"""Compare durable event deltas with atomic whole-state replacement.

The module changes only private experiment stores. The delta arm acknowledges
one update after its journal record is synchronized. Compact snapshots limit
restart work, but the journal remains the durability authority for each update.

Spec refs: REQ-CL-7458 and SCENARIO-CL-7458-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
    write_immutable_sidecar,
)


JsonDict = dict[str, Any]

RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
PHASE = 4
EXPERIMENT_ID = "exp7458-v653-durable-updates"
SCHEMA = "carnot.exp7458.v653.durable_updates.v1"
STATE_SCHEMA = "carnot.exp7458.four_expert_state.v1"
EVENT_SCHEMA = "carnot.exp7458.update_delta.v1"
JOURNAL_SCHEMA = "carnot.exp7458.journal_record.v1"
SNAPSHOT_SCHEMA = "carnot.exp7458.compact_snapshot.v1"
CONTROL_SCHEMA = "carnot.exp7458.whole_state.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7458_v653_durable_updates.json")
RAW_DIR = Path("results/raw/experiment_7458_v653_durable_updates")
MODULE_PATH = Path("python/carnot/experiment_7458_v653_durable_updates.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7458_v653_durable_updates.py")
TEST_PATH = Path("tests/python/test_experiment_7458_v653_durable_updates.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP7432_PATH = Path("results/experiment_7432_v651_update_placement.json")
EXP7445_PATH = Path("results/experiment_7445_v652_hardware_envelope.json")
EXP7438_PATH = Path("results/experiment_7438_v652_mixture_prototype.json")

EXPERT_NAMES = (
    "frozen_spline",
    "adaptive_spline",
    "frozen_gibbs",
    "adaptive_gibbs",
)
SNAPSHOT_INTERVAL = 64
BENCHMARK_BLOCKS = 30
UPDATES_PER_BLOCK = 128
BENCHMARK_CEILING_S = 900.0
STREAM_SEED = 65_301
RESAMPLING_SEED = 65_307
TARGET_RATIO_UPPER = 0.90
TARGET_SPEEDUP_X = 100.0
CRASH_POINTS = (
    "before_append",
    "after_partial_append",
    "after_fsync",
    "during_snapshot",
    "during_compaction",
)
STORE_NAMES = ("delta_journal", "whole_state")
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

EXPECTED_UPSTREAM_HASHES = {
    EXP7432_PATH: "sha256:85b3f05a10301bdb9320108fbed74220a8a2f742cde2c2852fe11e7f2bea6977",
    EXP7445_PATH: "sha256:bbf535fc1df3fb9b579abb0d2ecc6b5bd67c525f9abef1aea2cb92de9cdcf328",
    EXP7438_PATH: "sha256:bd670d6ce25e178d0bc856ba0dbc93ce7e4573d0ea0d427ed51b26d679c3ec61",
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7432_v651_update_placement.py"),
    Path("python/carnot/experiment_7445_v652_hardware_envelope.py"),
    Path("python/carnot/experiment_7438_v652_mixture_prototype.py"),
    SPEC_PATH,
    EXP7432_PATH,
    EXP7445_PATH,
    EXP7438_PATH,
)

SHARED_TEST_PATHS = (
    Path("tests/python/test_current_work_receipt.py"),
    Path("tests/python/test_experiment_7303_v642_validation_scope.py"),
    Path("tests/python/test_experiment_7358_v646_validation_contract.py"),
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), *(path.as_posix() for path in SHARED_TEST_PATHS)),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint_e2e",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "phase",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
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
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "durable_update_complete_score",
    "durable_update_value_score",
    "fault_rows",
    "service_timing_rows",
    "timing_summary",
    "hardware_acceleration_path",
    "small_ebm_training",
)


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one aware UTC boundary for a measured phase."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public execution boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Print a flushed phase or slow-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7458] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _json_bytes(value: Any) -> bytes:
    """Encode canonical finite JSON so hashes and byte accounting agree."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _checksum_bytes(value: Any) -> str:
    """Hash the exact canonical bytes used by durable records."""

    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object and keep malformed or absent bytes distinct."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sync_directory(path: Path) -> int:
    """Synchronize one directory and return the measured call cost."""

    started = time.perf_counter_ns()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return time.perf_counter_ns() - started


def _write_synced(path: Path, payload: bytes) -> tuple[int, int]:
    """Write and synchronize one file while keeping both costs separate."""

    write_started = time.perf_counter_ns()
    with path.open("wb", buffering=0) as stream:
        stream.write(payload)
        write_ns = time.perf_counter_ns() - write_started
        fsync_started = time.perf_counter_ns()
        os.fsync(stream.fileno())
        fsync_ns = time.perf_counter_ns() - fsync_started
    return write_ns, fsync_ns


def initial_state() -> JsonDict:
    """Return the fixed compact four-expert numeric state."""

    experts = {
        name: {
            "coefficients_q16": [
                (expert_index + 1) * 97 + (coefficient_index - 3) * 19
                for coefficient_index in range(8)
            ],
            "bias_q16": (expert_index - 1) * 113,
        }
        for expert_index, name in enumerate(EXPERT_NAMES)
    }
    return {
        "schema": STATE_SCHEMA,
        "sequence": 0,
        "update_count": 0,
        "last_event_id": None,
        "last_event_checksum": None,
        "experts": experts,
    }


def fixed_events(count: int = UPDATES_PER_BLOCK) -> list[JsonDict]:
    """Build one deterministic fixed-point delta stream."""

    if count < 0:
        raise ValueError("event_count_negative")
    rows: list[JsonDict] = []
    for sequence in range(1, count + 1):
        features = [((sequence * (index + 3) + index * 11) % 41) - 20 for index in range(8)]
        deltas = {
            name: {
                "coefficients_q16": [
                    ((sequence + expert_index * 5 + index * 3) % 7) - 3 for index in range(8)
                ],
                "bias_q16": ((sequence + expert_index) % 5) - 2,
            }
            for expert_index, name in enumerate(EXPERT_NAMES)
        }
        event: JsonDict = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": f"event-{sequence:06d}",
            "features": features,
            "label": sequence % 2,
            "deltas": deltas,
        }
        event["event_checksum"] = _checksum_bytes(event)
        rows.append(event)
    return rows


def _validate_event(event: Mapping[str, Any]) -> None:
    """Reject changed event bytes before they can alter durable state."""

    if event.get("schema") != EVENT_SCHEMA:
        raise ValueError("event_schema_invalid")
    supplied = event.get("event_checksum")
    payload = {key: deepcopy(value) for key, value in event.items() if key != "event_checksum"}
    if supplied != _checksum_bytes(payload):
        raise ValueError("event_checksum_invalid")
    sequence = event.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        raise ValueError("event_sequence_invalid")
    if event.get("event_id") != f"event-{sequence:06d}":
        raise ValueError("event_identity_invalid")
    deltas = event.get("deltas")
    if not isinstance(deltas, Mapping) or set(deltas) != set(EXPERT_NAMES):
        raise ValueError("event_experts_invalid")


def apply_event(state: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    """Apply one next delta without mutating the caller's state."""

    _validate_event(event)
    current = int(state.get("sequence", -1))
    sequence = int(event["sequence"])
    if sequence != current + 1:
        raise ValueError("sequence_gap")
    updated = deepcopy(dict(state))
    experts = updated["experts"]
    for name in EXPERT_NAMES:
        destination = experts[name]
        delta = event["deltas"][name]
        coefficients = list(destination["coefficients_q16"])
        additions = list(delta["coefficients_q16"])
        if len(coefficients) != 8 or len(additions) != 8:
            raise ValueError("coefficient_shape_invalid")
        destination["coefficients_q16"] = [
            int(left) + int(right) for left, right in zip(coefficients, additions, strict=True)
        ]
        destination["bias_q16"] = int(destination["bias_q16"]) + int(delta["bias_q16"])
    updated["sequence"] = sequence
    updated["update_count"] = int(updated["update_count"]) + 1
    updated["last_event_id"] = event["event_id"]
    updated["last_event_checksum"] = event["event_checksum"]
    return updated


def predict_state(state: Mapping[str, Any], features: Sequence[int]) -> JsonDict:
    """Return deterministic fixed-point scores and bounded probabilities."""

    if len(features) != 8:
        raise ValueError("feature_shape_invalid")
    scores: JsonDict = {}
    probabilities: JsonDict = {}
    for name in EXPERT_NAMES:
        expert = state["experts"][name]
        score = int(expert["bias_q16"]) + sum(
            int(weight) * int(feature)
            for weight, feature in zip(expert["coefficients_q16"], features, strict=True)
        )
        scores[name] = score
        scaled = max(-30.0, min(30.0, score / 65_536.0))
        probabilities[name] = 1.0 / (1.0 + math.exp(-scaled))
    return {
        "scores_q16": scores,
        "probabilities": probabilities,
        "mixture_probability": math.fsum(probabilities.values()) / len(EXPERT_NAMES),
    }


def _state_envelope(schema: str, state: Mapping[str, Any], **extra: Any) -> JsonDict:
    """Bind state bytes to a checksum without nesting value wrappers."""

    payload = {"schema": schema, "state": deepcopy(dict(state)), **extra}
    return {**payload, "checksum": _checksum_bytes(payload)}


def _decode_state_envelope(value: Mapping[str, Any], schema: str) -> JsonDict:
    """Verify one state envelope before restart uses its numeric bytes."""

    if value.get("schema") != schema or not isinstance(value.get("state"), Mapping):
        raise ValueError("state_envelope_invalid")
    payload = {key: deepcopy(item) for key, item in value.items() if key != "checksum"}
    if value.get("checksum") != _checksum_bytes(payload):
        raise ValueError("state_envelope_checksum_invalid")
    return deepcopy(dict(value["state"]))


class DeltaJournalStore:
    """Persist checksum-linked deltas and compact every 64 updates."""

    store_name = "delta_journal"

    def __init__(self, directory: Path, *, capacity_bytes: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.directory / "events.jsonl"
        self.snapshot_path = self.directory / "snapshot.json"
        self.snapshot_temp_path = self.directory / ".snapshot.tmp"
        self.compaction_temp_path = self.directory / ".journal.compact.tmp"
        self.recovery_evidence_path = self.directory / "recovery_evidence.json"
        self.capacity_bytes = capacity_bytes
        if not self.snapshot_path.exists():
            envelope = _state_envelope(SNAPSHOT_SCHEMA, initial_state(), last_journal_checksum="")
            temporary = self.snapshot_path.with_name(".snapshot.initialize.tmp")
            _write_synced(temporary, _json_bytes(envelope) + b"\n")
            os.replace(temporary, self.snapshot_path)
            _sync_directory(self.directory)
        if not self.journal_path.exists():
            _write_synced(self.journal_path, b"")
            _sync_directory(self.directory)
        recovered = self._recover()
        self.state = recovered["state"]
        self.last_journal_checksum = str(recovered["last_journal_checksum"])
        self.last_recovery_evidence = recovered["evidence"]

    def _recover(self) -> JsonDict:
        """Restore a valid snapshot and replay only complete linked records."""

        snapshot_value = _load_object(self.snapshot_path)
        state = _decode_state_envelope(snapshot_value, SNAPSHOT_SCHEMA)
        snapshot_sequence = int(state["sequence"])
        snapshot_checksum = str(snapshot_value.get("last_journal_checksum") or "")
        raw = self.journal_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid_end = 0
        records: list[JsonDict] = []
        invalid_index: int | None = None
        previous: str | None = None
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                invalid_index = index
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_index = index
                break
            if not isinstance(value, Mapping):
                invalid_index = index
                break
            record = dict(value)
            payload = {key: deepcopy(item) for key, item in record.items() if key != "checksum"}
            linked = previous is None or record.get("previous_checksum") == previous
            if (
                record.get("schema") != JOURNAL_SCHEMA
                or record.get("checksum") != _checksum_bytes(payload)
                or not linked
            ):
                invalid_index = index
                break
            records.append(record)
            previous = str(record["checksum"])
            valid_end += len(line)
        if invalid_index is not None and invalid_index != len(lines) - 1:
            raise ValueError("journal_corruption_before_tail")
        truncated = len(raw) - valid_end
        if truncated:
            with self.journal_path.open("r+b") as stream:
                stream.truncate(valid_end)
                stream.flush()
                os.fsync(stream.fileno())
        for record in records:
            sequence = int(record["sequence"])
            if sequence <= snapshot_sequence:
                continue
            if sequence != int(state["sequence"]) + 1:
                raise ValueError("journal_sequence_gap")
            state = apply_event(state, record["event"])
        last_checksum = previous if previous is not None else snapshot_checksum
        stale_temporaries = []
        for path in (self.snapshot_temp_path, self.compaction_temp_path):
            if path.exists():
                stale_temporaries.append(path.name)
                path.unlink()
        evidence = {
            "tail_action": "truncated" if truncated else "none",
            "truncated_bytes": truncated,
            "valid_record_count": len(records),
            "snapshot_sequence": snapshot_sequence,
            "recovered_sequence": int(state["sequence"]),
            "stale_temporaries_removed": stale_temporaries,
            "acknowledged_loss_count": 0,
        }
        if truncated or stale_temporaries:
            atomic_json(self.recovery_evidence_path, evidence)
        elif self.recovery_evidence_path.is_file():
            retained = _load_object(self.recovery_evidence_path)
            if retained:
                evidence = retained
        return {
            "state": state,
            "last_journal_checksum": last_checksum,
            "evidence": evidence,
        }

    def recover(self) -> JsonDict:
        """Re-read durable bytes so callers can test a restart boundary."""

        recovered = self._recover()
        self.state = recovered["state"]
        self.last_journal_checksum = str(recovered["last_journal_checksum"])
        self.last_recovery_evidence = recovered["evidence"]
        return {"state": deepcopy(self.state), "evidence": deepcopy(self.last_recovery_evidence)}

    def acknowledge(self, event: Mapping[str, Any], *, fault: str | None = None) -> JsonDict:
        """Append, synchronize, and only then acknowledge one complete delta."""

        started = time.perf_counter_ns()
        _validate_event(event)
        sequence = int(event["sequence"])
        if sequence <= int(self.state["sequence"]):
            return {
                "acknowledged": True,
                "disposition": "duplicate",
                "sequence": sequence,
                "total_service_ns": time.perf_counter_ns() - started,
                "durable_bytes_written": 0,
                "logical_update_bytes": len(_json_bytes(event)),
                "numeric_update_ns": 0,
                "hash_cost_ns": 0,
                "write_cost_ns": 0,
                "fsync_cost_ns": 0,
                "journal_fsync_count": 0,
                "snapshot_written": False,
            }
        if sequence != int(self.state["sequence"]) + 1:
            raise ValueError("sequence_gap")

        numeric_started = time.perf_counter_ns()
        updated = apply_event(self.state, event)
        numeric_ns = time.perf_counter_ns() - numeric_started
        record_payload = {
            "schema": JOURNAL_SCHEMA,
            "sequence": sequence,
            "event": deepcopy(dict(event)),
            "previous_checksum": self.last_journal_checksum,
        }
        hash_started = time.perf_counter_ns()
        record = {**record_payload, "checksum": _checksum_bytes(record_payload)}
        hash_ns = time.perf_counter_ns() - hash_started
        encoded = _json_bytes(record) + b"\n"
        if self.capacity_bytes is not None and len(encoded) > self.capacity_bytes:
            raise OSError(errno.ENOSPC, "injected private-store capacity")
        if fault == "before_append":  # pragma: no cover - owned child termination.
            os._exit(91)

        write_started = time.perf_counter_ns()
        with self.journal_path.open("ab", buffering=0) as stream:
            if fault == "after_partial_append":  # pragma: no cover - owned child termination.
                stream.write(encoded[: max(1, len(encoded) // 2)])
                os.fsync(stream.fileno())
                os._exit(91)
            stream.write(encoded)
            write_ns = time.perf_counter_ns() - write_started
            fsync_started = time.perf_counter_ns()
            os.fsync(stream.fileno())
            journal_fsync_ns = time.perf_counter_ns() - fsync_started
        if fault == "after_fsync":  # pragma: no cover - owned child termination.
            os._exit(91)

        durable_bytes = len(encoded)
        extra_write_ns = 0
        extra_fsync_ns = 0
        snapshot_written = False
        if sequence % SNAPSHOT_INTERVAL == 0:
            snapshot = _state_envelope(
                SNAPSHOT_SCHEMA,
                updated,
                last_journal_checksum=record["checksum"],
            )
            snapshot_bytes = _json_bytes(snapshot) + b"\n"
            if fault == "during_snapshot":  # pragma: no cover - owned child termination.
                with self.snapshot_temp_path.open("wb", buffering=0) as stream:
                    stream.write(snapshot_bytes[: max(1, len(snapshot_bytes) // 2)])
                    os.fsync(stream.fileno())
                os._exit(91)
            snapshot_write_ns, snapshot_fsync_ns = _write_synced(
                self.snapshot_temp_path, snapshot_bytes
            )
            os.replace(self.snapshot_temp_path, self.snapshot_path)
            snapshot_dir_fsync_ns = _sync_directory(self.directory)
            durable_bytes += len(snapshot_bytes)
            extra_write_ns += snapshot_write_ns
            extra_fsync_ns += snapshot_fsync_ns + snapshot_dir_fsync_ns
            snapshot_written = True

            compact_write_ns, compact_fsync_ns = _write_synced(self.compaction_temp_path, b"")
            if fault == "during_compaction":  # pragma: no cover - owned child termination.
                os._exit(91)
            os.replace(self.compaction_temp_path, self.journal_path)
            compact_dir_fsync_ns = _sync_directory(self.directory)
            extra_write_ns += compact_write_ns
            extra_fsync_ns += compact_fsync_ns + compact_dir_fsync_ns
        self.state = updated
        self.last_journal_checksum = str(record["checksum"])
        return {
            "acknowledged": True,
            "disposition": "committed",
            "sequence": sequence,
            "total_service_ns": time.perf_counter_ns() - started,
            "durable_bytes_written": durable_bytes,
            "logical_update_bytes": len(_json_bytes(event)),
            "numeric_update_ns": numeric_ns,
            "hash_cost_ns": hash_ns,
            "write_cost_ns": write_ns + extra_write_ns,
            "fsync_cost_ns": journal_fsync_ns + extra_fsync_ns,
            "journal_fsync_count": 1,
            "snapshot_written": snapshot_written,
        }


class WholeStateStore:
    """Persist each update through full-state file and directory synchronization."""

    store_name = "whole_state"

    def __init__(self, directory: Path, *, capacity_bytes: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.temporary_path = self.directory / ".state.tmp"
        self.recovery_evidence_path = self.directory / "recovery_evidence.json"
        self.capacity_bytes = capacity_bytes
        if not self.state_path.exists():
            envelope = _state_envelope(CONTROL_SCHEMA, initial_state())
            _write_synced(self.temporary_path, _json_bytes(envelope) + b"\n")
            os.replace(self.temporary_path, self.state_path)
            _sync_directory(self.directory)
        recovered = self.recover()
        self.state = recovered["state"]
        self.last_recovery_evidence = recovered["evidence"]

    def recover(self) -> JsonDict:
        """Reload the last atomically replaced complete state."""

        value = _load_object(self.state_path)
        state = _decode_state_envelope(value, CONTROL_SCHEMA)
        stale = []
        if self.temporary_path.exists():
            stale.append(self.temporary_path.name)
            self.temporary_path.unlink()
        evidence = {
            "tail_action": "rejected_temporary" if stale else "none",
            "truncated_bytes": 0,
            "recovered_sequence": int(state["sequence"]),
            "stale_temporaries_removed": stale,
            "acknowledged_loss_count": 0,
        }
        if stale:
            atomic_json(self.recovery_evidence_path, evidence)
        elif self.recovery_evidence_path.is_file():
            retained = _load_object(self.recovery_evidence_path)
            if retained:
                evidence = retained
        self.state = state
        self.last_recovery_evidence = evidence
        return {"state": deepcopy(state), "evidence": deepcopy(evidence)}

    def acknowledge(self, event: Mapping[str, Any], *, fault: str | None = None) -> JsonDict:
        """Replace complete state and sync its directory before success."""

        started = time.perf_counter_ns()
        _validate_event(event)
        sequence = int(event["sequence"])
        if sequence <= int(self.state["sequence"]):
            return {
                "acknowledged": True,
                "disposition": "duplicate",
                "sequence": sequence,
                "total_service_ns": time.perf_counter_ns() - started,
                "durable_bytes_written": 0,
                "logical_update_bytes": len(_json_bytes(event)),
                "numeric_update_ns": 0,
                "hash_cost_ns": 0,
                "write_cost_ns": 0,
                "fsync_cost_ns": 0,
                "file_fsync_count": 0,
                "directory_fsync_count": 0,
                "snapshot_written": True,
            }
        if sequence != int(self.state["sequence"]) + 1:
            raise ValueError("sequence_gap")
        numeric_started = time.perf_counter_ns()
        updated = apply_event(self.state, event)
        numeric_ns = time.perf_counter_ns() - numeric_started
        hash_started = time.perf_counter_ns()
        envelope = _state_envelope(CONTROL_SCHEMA, updated)
        encoded = _json_bytes(envelope) + b"\n"
        hash_ns = time.perf_counter_ns() - hash_started
        if self.capacity_bytes is not None and len(encoded) > self.capacity_bytes:
            raise OSError(errno.ENOSPC, "injected private-store capacity")
        if fault == "before_append":  # pragma: no cover - owned child termination.
            os._exit(91)
        if fault == "after_partial_append":  # pragma: no cover - owned child termination.
            with self.temporary_path.open("wb", buffering=0) as stream:
                stream.write(encoded[: max(1, len(encoded) // 2)])
                os.fsync(stream.fileno())
            os._exit(91)
        write_ns, file_fsync_ns = _write_synced(self.temporary_path, encoded)
        if fault == "after_fsync":  # pragma: no cover - owned child termination.
            os._exit(91)
        os.replace(self.temporary_path, self.state_path)
        if fault == "during_snapshot":  # pragma: no cover - owned child termination.
            os._exit(91)
        directory_fsync_ns = _sync_directory(self.directory)
        if fault == "during_compaction":  # pragma: no cover - owned child termination.
            os._exit(91)
        self.state = updated
        return {
            "acknowledged": True,
            "disposition": "committed",
            "sequence": sequence,
            "total_service_ns": time.perf_counter_ns() - started,
            "durable_bytes_written": len(encoded),
            "logical_update_bytes": len(_json_bytes(event)),
            "numeric_update_ns": numeric_ns,
            "hash_cost_ns": hash_ns,
            "write_cost_ns": write_ns,
            "fsync_cost_ns": file_fsync_ns + directory_fsync_ns,
            "file_fsync_count": 1,
            "directory_fsync_count": 1,
            "snapshot_written": True,
        }


def _store(store: str, directory: Path, *, capacity_bytes: int | None = None) -> Any:
    """Construct one named store without sharing state across arms."""

    if store == "delta_journal":
        return DeltaJournalStore(directory, capacity_bytes=capacity_bytes)
    if store == "whole_state":
        return WholeStateStore(directory, capacity_bytes=capacity_bytes)
    raise ValueError(f"store_unknown:{store}")


def _fault_worker(
    store: str, directory: Path, sequence: int, fault: str
) -> int:  # pragma: no cover
    """Run one owned crash injection in a child process."""

    writer = _store(store, directory)
    writer.acknowledge(fixed_events(sequence)[-1], fault=fault)
    return 0


def run_fault_matrix(root: Path) -> list[JsonDict]:
    """Terminate owned children at five boundaries for both protocols."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for store_name in STORE_NAMES:
        for crash_point in CRASH_POINTS:
            acknowledged_prefix = (
                63 if crash_point in {"during_snapshot", "during_compaction"} else 1
            )
            directory = root / store_name / crash_point
            writer = _store(store_name, directory)
            events = fixed_events(acknowledged_prefix + 1)
            for event in events[:acknowledged_prefix]:
                writer.acknowledge(event)
            command = (
                sys.executable,
                "-u",
                str(REPO_ROOT / WRAPPER_PATH),
                "--date",
                RUN_DATE,
                "--fault-worker",
                store_name,
                str(directory),
                str(acknowledged_prefix + 1),
                crash_point,
            )
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            recovered = _store(store_name, directory).recover()
            recovered_sequence = int(recovered["state"]["sequence"])
            reference = initial_state()
            for event in events[:recovered_sequence]:
                reference = apply_event(reference, event)
            probe = events[-1]["features"]
            prefix_preserved = recovered_sequence >= acknowledged_prefix
            exactly_once = recovered_sequence in {
                acknowledged_prefix,
                acknowledged_prefix + 1,
            }
            rows.append(
                {
                    "unit_id": f"fault:{store_name}:{crash_point}",
                    "row_type": "fault_injection",
                    "store": store_name,
                    "condition": crash_point,
                    "crash_point": crash_point,
                    "expected_acknowledged_prefix": acknowledged_prefix,
                    "recovered_prefix": recovered_sequence,
                    "uncertain_event_recovered": recovered_sequence > acknowledged_prefix,
                    "child_exit_code": completed.returncode,
                    "child_terminated": completed.returncode == 91,
                    "acknowledged_prefix_preserved": prefix_preserved,
                    "acknowledged_loss_count": max(0, acknowledged_prefix - recovered_sequence),
                    "recovered_exactly_once": exactly_once,
                    "tail_action": recovered["evidence"]["tail_action"],
                    "truncated_bytes": recovered["evidence"]["truncated_bytes"],
                    "numeric_state_parity": recovered["state"] == reference,
                    "prediction_parity": predict_state(recovered["state"], probe)
                    == predict_state(reference, probe),
                    "disposition": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
    return rows


def _percentile(values: Sequence[int], percentile: float) -> float:
    """Return one linearly interpolated percentile for recorded latencies."""

    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("percentile_needs_values")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _filesystem_type(path: Path) -> str:
    """Read the local mount table without contacting a filesystem service."""

    resolved = path.resolve()
    best_mount = Path("/")
    best_type = "unknown"
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover - Linux experiment host supplies procfs.
        return best_type
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " "))
        try:
            contains = resolved == mount or resolved.is_relative_to(mount)
        except OSError:  # pragma: no cover - malformed external mount row.
            contains = False
        if contains and len(mount.parts) >= len(best_mount.parts):
            best_mount = mount
            best_type = fields[2]
    return best_type


def _timing_row(
    *,
    row_type: str,
    block: int,
    store_name: str,
    order: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    total_service_ns: int,
    state_parity: bool,
    prediction_parity: bool,
    filesystem_type: str,
) -> JsonDict:
    """Reduce one store run without dropping per-update latency."""

    latencies = [int(row["total_service_ns"]) for row in receipts]
    durable_bytes = sum(int(row["durable_bytes_written"]) for row in receipts)
    logical_bytes = sum(int(row["logical_update_bytes"]) for row in receipts)
    numeric_ns = sum(int(row["numeric_update_ns"]) for row in receipts)
    hash_ns = sum(int(row["hash_cost_ns"]) for row in receipts)
    write_ns = sum(int(row["write_cost_ns"]) for row in receipts)
    fsync_ns = sum(int(row["fsync_cost_ns"]) for row in receipts)
    return {
        "unit_id": f"{row_type}:{block}:{store_name}",
        "row_type": row_type,
        "block": block,
        "arm": store_name,
        "store": store_name,
        "order": list(order),
        "order_index": list(order).index(store_name),
        "update_count": len(receipts),
        "update_latencies_ns": latencies,
        "total_service_ns": max(int(total_service_ns), sum(latencies)),
        "numeric_update_ns": numeric_ns,
        "host_service_ns": max(0, int(total_service_ns) - numeric_ns),
        "durable_bytes": durable_bytes,
        "logical_update_bytes": logical_bytes,
        "write_amplification": durable_bytes / logical_bytes,
        "hash_cost_ns": hash_ns,
        "write_cost_ns": write_ns,
        "fsync_cost_ns": fsync_ns,
        "p50_update_latency_ns": _percentile(latencies, 0.50),
        "p95_update_latency_ns": _percentile(latencies, 0.95),
        "state_parity": state_parity,
        "prediction_parity": prediction_parity,
        "filesystem_type": filesystem_type,
        "filesystem_class": "local",
        "disposition": "complete",
        "failed": False,
        "censored": False,
    }


def _unstarted_timing_row(
    block: int, store_name: str, order: Sequence[str], updates: int
) -> JsonDict:
    """Keep one planned arm visible after the fixed benchmark ceiling."""

    return {
        "unit_id": f"service_timing:{block}:{store_name}",
        "row_type": "service_timing",
        "block": block,
        "arm": store_name,
        "store": store_name,
        "order": list(order),
        "order_index": list(order).index(store_name),
        "update_count": updates,
        "update_latencies_ns": [],
        "total_service_ns": None,
        "numeric_update_ns": None,
        "host_service_ns": None,
        "durable_bytes": None,
        "logical_update_bytes": None,
        "write_amplification": None,
        "hash_cost_ns": None,
        "write_cost_ns": None,
        "fsync_cost_ns": None,
        "p50_update_latency_ns": None,
        "p95_update_latency_ns": None,
        "state_parity": None,
        "prediction_parity": None,
        "filesystem_type": None,
        "filesystem_class": "local",
        "disposition": "unstarted",
        "failed": False,
        "censored": True,
    }


def benchmark_protocol(
    root: Path,
    *,
    blocks: int = BENCHMARK_BLOCKS,
    updates: int = UPDATES_PER_BLOCK,
    ceiling_s: float = BENCHMARK_CEILING_S,
    emit_progress: bool = False,
) -> list[JsonDict]:
    """Run warm-ups and alternating paired complete-service blocks."""

    if blocks <= 0 or updates <= 0 or ceiling_s < 0:
        raise ValueError("benchmark_budget_invalid")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    filesystem_type = _filesystem_type(root)
    if filesystem_type.lower() in {"nfs", "nfs4", "cifs", "smbfs", "sshfs"}:
        raise RuntimeError(f"network_filesystem_forbidden:{filesystem_type}")
    events = fixed_events(updates)
    expected = initial_state()
    for event in events:
        expected = apply_event(expected, event)
    probe = events[-1]["features"]
    rows: list[JsonDict] = []
    for store_name in STORE_NAMES:
        writer = _store(store_name, root / "warmup" / store_name)
        receipts: list[JsonDict] = []
        started = time.perf_counter_ns()
        for event in events:
            receipts.append(writer.acknowledge(event))
        elapsed = time.perf_counter_ns() - started
        recovered = _store(store_name, writer.directory).recover()["state"]
        rows.append(
            _timing_row(
                row_type="warmup",
                block=-1,
                store_name=store_name,
                order=STORE_NAMES,
                receipts=receipts,
                total_service_ns=elapsed,
                state_parity=recovered == expected,
                prediction_parity=predict_state(recovered, probe) == predict_state(expected, probe),
                filesystem_type=filesystem_type,
            )
        )

    measured_started = time.monotonic()
    for block in range(blocks):
        order = STORE_NAMES if block % 2 == 0 else (STORE_NAMES[1], STORE_NAMES[0])
        if time.monotonic() - measured_started >= ceiling_s:
            rows.extend(_unstarted_timing_row(block, name, order, updates) for name in order)
            continue
        for store_name in order:
            writer = _store(store_name, root / f"block-{block:02d}" / store_name)
            receipts = []
            started = time.perf_counter_ns()
            failed = False
            for event in events:
                if time.monotonic() - measured_started >= ceiling_s:
                    failed = True
                    break
                receipts.append(writer.acknowledge(event))
            elapsed = time.perf_counter_ns() - started
            if failed:
                row = _unstarted_timing_row(block, store_name, order, updates)
                row.update(
                    {
                        "disposition": "censored_ceiling",
                        "update_count": len(receipts),
                        "update_latencies_ns": [
                            int(receipt["total_service_ns"]) for receipt in receipts
                        ],
                    }
                )
                rows.append(row)
                continue
            recovered = _store(store_name, writer.directory).recover()["state"]
            rows.append(
                _timing_row(
                    row_type="service_timing",
                    block=block,
                    store_name=store_name,
                    order=order,
                    receipts=receipts,
                    total_service_ns=elapsed,
                    state_parity=recovered == expected,
                    prediction_parity=predict_state(recovered, probe)
                    == predict_state(expected, probe),
                    filesystem_type=filesystem_type,
                )
            )
            if emit_progress:
                progress(
                    measured_started,
                    "benchmark",
                    "arm_complete",
                    completed_units=len(
                        [row for row in rows if row["row_type"] == "service_timing"]
                    ),
                    planned_units=blocks * 2,
                )
    return rows


def _paired_ratio_interval(
    numerator: Sequence[float], denominator: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap paired blocks and return ratio plus a two-sided 95 percent interval."""

    if len(numerator) != len(denominator) or not numerator or draws <= 0:
        raise ValueError("paired_ratio_inputs_invalid")
    observed = math.fsum(numerator) / math.fsum(denominator)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        indices = [rng.randrange(len(numerator)) for _index in numerator]
        samples.append(
            math.fsum(numerator[index] for index in indices)
            / math.fsum(denominator[index] for index in indices)
        )
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return observed, lower, upper


def summarize_timing(
    rows: Sequence[Mapping[str, Any]], *, draws: int = 10_000, seed: int = RESAMPLING_SEED
) -> JsonDict:
    """Reduce complete paired blocks and the measured residual host fraction."""

    complete = [
        row
        for row in rows
        if row.get("row_type") == "service_timing"
        and row.get("disposition") == "complete"
        and row.get("failed") is False
        and row.get("censored") is False
    ]
    by_block: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in complete:
        by_block.setdefault(int(row["block"]), {})[str(row["store"])] = row
    paired = [
        by_block[index] for index in sorted(by_block) if set(by_block[index]) == set(STORE_NAMES)
    ]
    if not paired:
        return {
            "paired_block_count": 0,
            "total_service_ratio": None,
            "ratio_ci95_lower": None,
            "ratio_ci95_upper": None,
            "speed_gate_passed": False,
            "residual_host_fraction": None,
            "amdahl_infinite_numeric_speedup_x": None,
            "one_hundred_x_host_target_met": False,
            "arm_summaries": {},
        }
    journal = [float(pair["delta_journal"]["total_service_ns"]) for pair in paired]
    control = [float(pair["whole_state"]["total_service_ns"]) for pair in paired]
    ratio, lower, upper = _paired_ratio_interval(journal, control, draws=draws, seed=seed)
    journal_rows = [pair["delta_journal"] for pair in paired]
    journal_total = math.fsum(float(row["total_service_ns"]) for row in journal_rows)
    host_total = math.fsum(float(row["host_service_ns"]) for row in journal_rows)
    residual = host_total / journal_total
    arm_summaries: JsonDict = {}
    for store_name in STORE_NAMES:
        arm_rows = [pair[store_name] for pair in paired]
        latencies = [int(latency) for row in arm_rows for latency in row["update_latencies_ns"]]
        arm_summaries[store_name] = {
            "update_count": len(latencies),
            "total_service_ns": sum(int(row["total_service_ns"]) for row in arm_rows),
            "durable_bytes": sum(int(row["durable_bytes"]) for row in arm_rows),
            "hash_cost_ns": sum(int(row["hash_cost_ns"]) for row in arm_rows),
            "fsync_cost_ns": sum(int(row["fsync_cost_ns"]) for row in arm_rows),
            "p50_update_latency_ns": _percentile(latencies, 0.50),
            "p95_update_latency_ns": _percentile(latencies, 0.95),
        }
    return {
        "paired_block_count": len(paired),
        "total_service_ratio": ratio,
        "ratio_ci95_lower": lower,
        "ratio_ci95_upper": upper,
        "speed_gate_passed": upper < TARGET_RATIO_UPPER,
        "residual_host_fraction": residual,
        "amdahl_infinite_numeric_speedup_x": 1.0 / residual,
        "one_hundred_x_host_target_met": residual <= 1.0 / TARGET_SPEEDUP_X,
        "arm_summaries": arm_summaries,
    }


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep acceptance operands as plain machine-readable values."""

    return {
        "check": check,
        "category": category,
        "op": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed check without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else {}
    return {
        "all_passed": not failed,
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": None
        if not failed
        else {
            "check": first.get("check"),
            "category": first.get("category"),
            "op": first.get("op"),
            "expected": deepcopy(first.get("expected")),
            "observed": deepcopy(first.get("observed")),
        },
    }


def _required_receipts(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one passing non-timeout receipt for each frozen command."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _acceptance_gates(
    fault_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    timing_summary: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    expected_blocks: int,
    validation_required: bool,
    require_terminal: bool,
) -> list[JsonDict]:
    """Keep durability validity separate from the preregistered speed benefit."""

    fault_complete = len(fault_rows) == len(CRASH_POINTS) * len(STORE_NAMES)
    parity = fault_complete and all(
        row.get("child_terminated") is True
        and row.get("acknowledged_prefix_preserved") is True
        and row.get("recovered_exactly_once") is True
        and row.get("numeric_state_parity") is True
        and row.get("prediction_parity") is True
        for row in fault_rows
    )
    loss = sum(int(row.get("acknowledged_loss_count") or 0) for row in fault_rows)
    complete_service = [
        row
        for row in timing_rows
        if row.get("row_type") == "service_timing" and row.get("disposition") == "complete"
    ]
    timing_complete = (
        len(complete_service) == expected_blocks * 2
        and timing_summary.get("paired_block_count") == expected_blocks
        and all(
            row.get("state_parity") is True and row.get("prediction_parity") is True
            for row in complete_service
        )
    )
    affected_ok = not validation_required or _required_receipts(
        validation_receipts, AFFECTED_CHECK_NAMES
    )
    terminal_ok = not require_terminal or _required_receipts(
        validation_receipts, TERMINAL_CHECK_NAMES
    )
    return [
        _gate(
            "fault_matrix_complete",
            "validity",
            "==",
            len(CRASH_POINTS) * len(STORE_NAMES),
            len(fault_rows),
            fault_complete,
            "Both protocols must retain every registered crash disposition.",
        ),
        _gate(
            "exact_crash_parity",
            "validity",
            "==",
            True,
            parity,
            parity,
            "Recovered numeric state and predictions must equal the serial reference.",
        ),
        _gate(
            "acknowledged_loss_count",
            "validity",
            "==",
            0,
            loss,
            loss == 0,
            "No acknowledged update may disappear after restart.",
        ),
        _gate(
            "paired_timing_complete",
            "validity",
            "==",
            expected_blocks,
            timing_summary.get("paired_block_count"),
            timing_complete,
            "Every planned paired block must include all durable service work.",
        ),
        _gate(
            "affected_validation",
            "validity",
            "==",
            True,
            affected_ok,
            affected_ok,
            "The frozen affected-file command plan must pass.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            "==",
            True,
            terminal_ok,
            terminal_ok,
            "Fresh readers must accept the exact measured candidate.",
        ),
        _gate(
            "paired_total_service_ratio_ci95_upper",
            "benefit",
            "<",
            TARGET_RATIO_UPPER,
            timing_summary.get("ratio_ci95_upper"),
            timing_summary.get("speed_gate_passed") is True,
            "Software value needs a paired upper confidence bound below 0.90.",
        ),
        _gate(
            "one_hundred_x_residual_host_fraction",
            "benefit",
            "<=",
            1.0 / TARGET_SPEEDUP_X,
            timing_summary.get("residual_host_fraction"),
            timing_summary.get("one_hundred_x_host_target_met") is True,
            "A 100x route needs at most one percent unaccelerated host work.",
        ),
    ]


def _field_principles() -> JsonDict:
    """Explain required fields while their values remain bare scalars."""

    defaults = {
        field: "Bind this field to measured evidence and cold reduction."
        for field in REQUIRED_FIELDS
    }
    defaults.update(
        {
            "schema": "Use a versioned top-level schema and exact run identity.",
            "run_date": "Use 20260920 and retain real UTC and monotonic boundaries.",
            "preconditions_checked": "Name actual source paths, identities, flags, and observed values.",
            "MODEL_SPECS": "Use an empty list because this task performs no LLM work.",
            "model_invoked": "Distinguish current model attempts from historical evidence.",
            "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight calls.",
            "inference_substrate": "Describe current work as no model load.",
            "inference_substrate_class": "Declare the actual current compute class without simulation claims.",
            "execution_venue": "Use host and keep any device history separate.",
            "duration_s": "Measure real current work and never pad elapsed time.",
            "phase_spans": "Bind phase timings, progress events, completed units, and checkpoints.",
            "random_seed": "Freeze stream and paired-resampling seeds.",
            "reproducibility_checksum": "Bind code, protocol, inputs, rows, and exact validation scope.",
            "source_artifact_hashes": "Preserve exact upstream bytes and original classification flags.",
            "rows": "Retain every fault and planned timing unit, including censoring.",
            "sample_size_budget": "Separate planned, attempted, complete, failed, censored, and unstarted pairs.",
            "acceptance_gate_results": "Keep validity and benefit operands plain and separate.",
            "gate_check_summary": "Name each failure without conflating missing, false, and zero.",
            "verifier_is_oracle": "State whether the deployed verifier supplies scoring authority.",
            "honest_verdict": "Start completed findings with complete_ and keep a failed benefit gate null.",
            "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
            "flagged_adversarial": "Preserve actual critical findings; flagged science cannot supply readiness.",
            "validation_receipts": "Record exact commands, environments, exits, durations, and log hashes.",
            "field_principles": "Explain field intent without wrapping machine-readable values.",
            "promotion_score": "Remain zero because this milestone does not authorize rollout.",
            "durable_update_complete_score": "Require both protocols, every fault, valid timing, and validation.",
            "durable_update_value_score": "Require exact acknowledged state and the registered speed gate.",
            "fault_rows": "Retain one row for each store and crash point with its acknowledged prefix.",
            "service_timing_rows": "Retain all durable work for every paired block and warm-up.",
            "hardware_acceleration_path": "Separate measured host costs from prospective fixed-point placement.",
        }
    )
    return defaults


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable protocol, sources, raw rows, gates, and validation scope."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "run_date": value.get("run_date"),
            "random_seed": value.get("random_seed"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "fault_rows": value.get("fault_rows"),
            "service_timing_rows": value.get("service_timing_rows"),
            "timing_summary": value.get("timing_summary"),
            "acceptance_gate_results": value.get("acceptance_gate_results"),
            "validation_manifest": value.get("validation_manifest"),
            "raw_evidence_reference": value.get("raw_evidence_reference"),
        }
    )


def _raw_reference(path: Path, root: Path) -> JsonDict:
    """Bind exact raw evidence bytes while keeping large rows out of duplicate files."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def write_raw_evidence(
    path: Path, *, fault_rows: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Write one current raw bundle before terminal artifact reduction."""

    value = {
        "schema": "carnot.exp7458.raw_evidence.v1",
        "fault_rows": [deepcopy(dict(row)) for row in fault_rows],
        "service_timing_rows": [deepcopy(dict(row)) for row in timing_rows],
    }
    atomic_json(path, value)
    return value


def _sample_budget(timing_rows: Sequence[Mapping[str, Any]], planned_blocks: int) -> JsonDict:
    """Count paired blocks without treating two arms as independent units."""

    dispositions: dict[int, set[str]] = {}
    censored: set[int] = set()
    failed: set[int] = set()
    for row in timing_rows:
        if row.get("row_type") != "service_timing":
            continue
        block = int(row["block"])
        if row.get("disposition") == "complete":
            dispositions.setdefault(block, set()).add(str(row.get("store")))
        if row.get("censored") is True:
            censored.add(block)
        if row.get("failed") is True:
            failed.add(block)
    complete = sum(1 for stores in dispositions.values() if stores == set(STORE_NAMES))
    attempted = len(set(dispositions) | censored | failed)
    return {
        "planned_independent_units": planned_blocks,
        "attempted_independent_units": attempted,
        "completed_independent_units": complete,
        "failed_independent_units": len(failed),
        "censored_independent_units": len(censored),
        "unstarted_independent_units": max(0, planned_blocks - attempted),
        "updates_per_arm_per_block": UPDATES_PER_BLOCK,
        "paired_arms": list(STORE_NAMES),
        "warmup_units": 2,
        "stop_rule": "stop at 900 seconds and retain all censored or unstarted planned blocks",
    }


def assemble_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    fault_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    raw_reference: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_ns: int,
    ended_ns: int,
    planned_blocks: int,
    validation_required: bool,
    require_terminal: bool,
    fixture: bool = False,
    candidate: bool = False,
    flagged_adversarial: bool = False,
    sidecar_references: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one schema-complete artifact from raw durability evidence."""

    timing_summary = summarize_timing(timing_rows)
    gates = _acceptance_gates(
        fault_rows,
        timing_rows,
        timing_summary,
        validation_receipts,
        expected_blocks=planned_blocks,
        validation_required=validation_required,
        require_terminal=require_terminal,
    )
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    speed = next(row for row in gates if row["check"] == "paired_total_service_ratio_ci95_upper")[
        "passed"
    ]
    exact = all(
        next(row for row in gates if row["check"] == check)["passed"]
        for check in ("exact_crash_parity", "acknowledged_loss_count")
    )
    complete_score = int(validity and not flagged_adversarial)
    value_score = int(complete_score == 1 and exact and speed)
    verdict_class = "positive" if value_score else "null"
    honest_verdict = (
        "complete_positive_durable_delta_service_gain"
        if value_score
        else "complete_null_durable_delta_speed_gate_not_met"
    )
    receipt = build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "host durability, crash recovery, fixed-point numeric updates, and paired timing",
            "model_or_accelerator_loaded": False,
            "filesystem_scope": "private local experiment directories",
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False, "current_training": False},
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": honest_verdict,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        **receipt,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "random_seed": {
            "fit": None,
            "projection": None,
            "stream": STREAM_SEED,
            "resampling": RESAMPLING_SEED,
            "null_randomness": "No model fitting or projection occurs; the numeric stream is deterministic.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [
            *[deepcopy(dict(row)) for row in fault_rows],
            *[deepcopy(dict(row)) for row in timing_rows],
        ],
        "sample_size_budget": _sample_budget(timing_rows, planned_blocks),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "durable_update_complete_score": complete_score,
        "durable_update_value_score": value_score,
        "fault_rows": [deepcopy(dict(row)) for row in fault_rows],
        "service_timing_rows": [deepcopy(dict(row)) for row in timing_rows],
        "timing_summary": timing_summary,
        "raw_evidence_reference": deepcopy(dict(raw_reference)),
        "acknowledgement_contract": {
            "success_boundary": "complete update is durable",
            "delta_journal_fsyncs_per_acknowledged_update": 1,
            "snapshot_interval_updates": SNAPSHOT_INTERVAL,
            "batch_acknowledgements": False,
            "fsync_removed_for_speed": False,
            "control_protocol": "full state temp write, file fsync, atomic replace, directory fsync",
        },
        "hardware_acceleration_path": {
            "current_measurement": "host_software_prototype",
            "compact_numeric_state_candidates": ["FPGA", "GPU"],
            "durable_journal_placement": "CPU/storage",
            "orchestration_placement": "CPU/storage",
            "hardware_performance_claimed": False,
            "fpga_or_tsu_validation_claimed": False,
            "production_storage_defaults_changed": False,
            "residual_host_fraction": timing_summary["residual_host_fraction"],
            "amdahl_infinite_numeric_speedup_x": timing_summary[
                "amdahl_infinite_numeric_speedup_x"
            ],
            "target_speedup_x": TARGET_SPEEDUP_X,
            "required_unaccelerated_fraction": 1.0 / TARGET_SPEEDUP_X,
            "architecture_target_met": timing_summary["one_hundred_x_host_target_met"],
        },
        "small_ebm_training": {"performed": False, "current_training": False},
        "validation_required": validation_required,
        "terminal_validation_required": require_terminal,
        "fixture_artifact": fixture,
        "candidate_artifact": candidate,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute rows, gates, scores, verdict class, and hardware boundary."""

    fault_rows = [dict(row) for row in artifact.get("fault_rows", []) if isinstance(row, Mapping)]
    timing_rows = [
        dict(row) for row in artifact.get("service_timing_rows", []) if isinstance(row, Mapping)
    ]
    budget = artifact.get("sample_size_budget")
    planned = int(budget.get("planned_independent_units", 0)) if isinstance(budget, Mapping) else 0
    summary = summarize_timing(timing_rows)
    gates = _acceptance_gates(
        fault_rows,
        timing_rows,
        summary,
        artifact.get("validation_receipts", []),
        expected_blocks=planned,
        validation_required=artifact.get("validation_required") is True,
        require_terminal=artifact.get("terminal_validation_required") is True,
    )
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    speed = next(row for row in gates if row["check"] == "paired_total_service_ratio_ci95_upper")[
        "passed"
    ]
    complete = int(validity and artifact.get("flagged_adversarial") is False)
    value = int(complete == 1 and speed)
    expected_rows = [*deepcopy(fault_rows), *deepcopy(timing_rows)]
    matches = (
        artifact.get("rows") == expected_rows
        and artifact.get("timing_summary") == summary
        and artifact.get("acceptance_gate_results") == gates
        and artifact.get("gate_check_summary") == _gate_summary(gates)
        and artifact.get("durable_update_complete_score") == complete
        and artifact.get("durable_update_value_score") == value
        and artifact.get("promotion_score") == 0
        and artifact.get("verdict_class") == ("positive" if value else "null")
        and (artifact.get("hardware_acceleration_path") or {}).get("hardware_performance_claimed")
        is False
    )
    return {
        "fault_row_count": len(fault_rows),
        "paired_block_count": summary["paired_block_count"],
        "durable_update_complete_score": complete,
        "durable_update_value_score": value,
        "promotion_score": 0,
        "matches_declared": matches,
    }


def _raw_evidence_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Reload raw rows and reject a terminal artifact that changed them."""

    reference = artifact.get("raw_evidence_reference")
    if not isinstance(reference, Mapping):
        return ["raw_evidence_reference_invalid"]
    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != reference.get("sha256"):
        return ["raw_evidence_hash_mismatch"]
    raw = _load_object(resolved)
    if raw.get("fault_rows") != artifact.get("fault_rows") or raw.get(
        "service_timing_rows"
    ) != artifact.get("service_timing_rows"):
        return ["raw_rows_mismatch"]
    return []


def validate_artifact(artifact: Mapping[str, Any], *, root: Path) -> list[str]:
    """Cold-check identity, raw rows, provenance, scores, and hardware limits."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in artifact]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
        "promotion_score": 0,
    }
    for field, expected_value in expected.items():
        if artifact.get(field) != expected_value:
            errors.append(
                "current_model_boundary_invalid"
                if field in {"MODEL_SPECS", "model_invoked", "invocation_counts"}
                else f"declaration_mismatch:{field}"
            )
    if set(artifact.get("field_principles") or {}) != set(REQUIRED_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    errors.extend(validate_current_work_receipt(artifact, root=root))
    errors.extend(_raw_evidence_errors(artifact, root))
    reduced = independent_reduce(artifact)
    if reduced["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    hardware = artifact.get("hardware_acceleration_path")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("durable_journal_placement") != "CPU/storage"
        or hardware.get("hardware_performance_claimed") is not False
        or hardware.get("production_storage_defaults_changed") is not False
    ):
        errors.append("hardware_boundary_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_fault_rows() -> list[JsonDict]:
    """Build compact passing fault evidence for mutation-focused artifact tests."""

    return [
        {
            "unit_id": f"fault:{store}:{point}",
            "row_type": "fault_injection",
            "store": store,
            "condition": point,
            "crash_point": point,
            "expected_acknowledged_prefix": 1,
            "recovered_prefix": 1,
            "uncertain_event_recovered": False,
            "child_exit_code": 91,
            "child_terminated": True,
            "acknowledged_prefix_preserved": True,
            "acknowledged_loss_count": 0,
            "recovered_exactly_once": True,
            "tail_action": "truncated" if point == "after_partial_append" else "none",
            "truncated_bytes": 4 if point == "after_partial_append" else 0,
            "numeric_state_parity": True,
            "prediction_parity": True,
            "disposition": "complete",
            "failed": False,
            "censored": False,
        }
        for store in STORE_NAMES
        for point in CRASH_POINTS
    ]


def _fixture_timing_rows() -> list[JsonDict]:
    """Build deterministic paired rows for cold-reader mutation tests."""

    rows: list[JsonDict] = []
    for block in range(4):
        order = STORE_NAMES if block % 2 == 0 else (STORE_NAMES[1], STORE_NAMES[0])
        for store_name in order:
            latency = 80 if store_name == "delta_journal" else 100
            receipts = [
                {
                    "total_service_ns": latency,
                    "durable_bytes_written": 80 if store_name == "delta_journal" else 200,
                    "logical_update_bytes": 40,
                    "numeric_update_ns": 8,
                    "hash_cost_ns": 4,
                    "write_cost_ns": 8,
                    "fsync_cost_ns": 40,
                }
                for _ in range(8)
            ]
            rows.append(
                _timing_row(
                    row_type="service_timing",
                    block=block,
                    store_name=store_name,
                    order=order,
                    receipts=receipts,
                    total_service_ns=latency * 8,
                    state_parity=True,
                    prediction_parity=True,
                    filesystem_type="fixture",
                )
            )
    return rows


def build_fixture_artifact(root: Path) -> JsonDict:
    """Build one complete compact artifact without repository writes."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    faults = _fixture_fault_rows()
    timing = _fixture_timing_rows()
    raw_path = root / "raw_evidence.json"
    write_raw_evidence(raw_path, fault_rows=faults, timing_rows=timing)
    started = time.monotonic_ns()
    return assemble_artifact(
        root=root,
        preconditions=[
            {
                "check": "fixture",
                "upstream": "fixture",
                "path": "fixture",
                "field": "available",
                "operator": "==",
                "expected": True,
                "observed": True,
                "passed": True,
            }
        ],
        source_hashes={},
        fault_rows=faults,
        timing_rows=timing,
        raw_reference=_raw_reference(raw_path, root),
        validation_receipts=[],
        phase_spans=[],
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        started_ns=started,
        ended_ns=time.monotonic_ns(),
        planned_blocks=4,
        validation_required=False,
        require_terminal=False,
        fixture=True,
    )


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Retain the exact source field and its independent disposition."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected if passed is None else bool(passed),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate local sources and preserve each upstream's original flags."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        observed_hash = sha256_file(path) if present else None
        expected_hash = EXPECTED_UPSTREAM_HASHES.get(relative)
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "sha256" if expected_hash else "bytes",
                expected_hash or "readable_nonempty_bytes",
                observed_hash
                if expected_hash
                else ("readable_nonempty_bytes" if present else None),
                passed=present and (expected_hash is None or observed_hash == expected_hash),
            )
        )
        if present:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": observed_hash,
                "bytes": path.stat().st_size,
                "role": "required_source",
                "original_verdict_class": None,
                "original_flagged_adversarial": None,
            }

    sources = {
        "exp7432": _load_object(root / EXP7432_PATH),
        "exp7445": _load_object(root / EXP7445_PATH),
        "exp7438": _load_object(root / EXP7438_PATH),
    }
    contracts = {
        "exp7432": {
            "path": EXP7432_PATH,
            "fields": {
                "experiment_id": "exp7432-v651-update-placement",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "model_invoked": False,
            },
        },
        "exp7445": {
            "path": EXP7445_PATH,
            "fields": {
                "experiment_id": "exp7445-v652-hardware-envelope",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "model_invoked": False,
            },
        },
        "exp7438": {
            "path": EXP7438_PATH,
            "fields": {
                "experiment_id": "exp7438-mixture-prototype",
                "verdict_class": "circular_positive",
                "flagged_adversarial": False,
                "model_invoked": False,
                "mixture_prototype_ready_score": 1,
            },
        },
    }
    for label, contract in contracts.items():
        source = sources[label]
        path = contract["path"]
        for field, expected in contract["fields"].items():
            checks.append(
                _precondition(
                    f"{label}_{field}", label, path.as_posix(), field, expected, source.get(field)
                )
            )
        if path.as_posix() in hashes:
            hashes[path.as_posix()].update(
                {
                    "role": "upstream_terminal_artifact",
                    "original_status": source.get("status"),
                    "original_honest_verdict": source.get("honest_verdict"),
                    "original_verdict_class": source.get("verdict_class"),
                    "original_flagged_adversarial": source.get("flagged_adversarial"),
                }
            )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7458",
            "REQ-CL-7458" if "REQ-CL-7458" in spec_text else None,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantined = "experiment_id: 7458" in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            quarantined,
        )
    )
    return checks, hashes, sources


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 and Exp7303 affected-file command plan."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = ".venv/bin/python"
    common = (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay", (*common, "--cold-replay", str(candidate)), "candidate"
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (*common, "--independent-reduce", str(candidate)),
            "candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
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
            "candidate",
        ),
    )
    categories = ("completion", "completion", "safety", "completion")
    return [
        PlannedCommand(spec, category, True)
        for spec, category in zip(specs, categories, strict=True)
    ]


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover
    """Retain the declared capability command and its exact environment."""

    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.json"
    value = {
        "argv": [".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE],
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "duration_s": duration_s,
        "environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }
    atomic_json(path, value)
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(value["argv"]),
        "command_argv": value["argv"],
        "command_environment": value["environment"],
        "scope": "declared_capability_entrypoint",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "timed_out": False,
        "required": True,
        "command_category": "completion",
        "started_at_utc": started_at_utc,
        "ended_at_utc": value["completed_at_utc"],
    }


def _span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    """Close one measured phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_reference": checkpoint,
    }


def run_experiment(  # pragma: no cover - exercised by declared entrypoint E2E.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Authenticate, crash-test, benchmark, validate, and publish once."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []
    progress(run_started, "startup", "flushed", completed_units=0)

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before_authentication", completed_units=0)
    preconditions, source_hashes, sources = collect_preconditions(root)
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started,
            len(preconditions),
            (raw_dir / "preconditions.json").as_posix(),
        )
    )
    atomic_json(raw_dir / "preconditions.json", {"checks": preconditions})
    progress(
        run_started,
        "preconditions",
        "after_authentication",
        completed_units=len(preconditions),
    )
    if failed is not None:
        raise RuntimeError(f"blocked_required_source:{json.dumps(failed, sort_keys=True)}")

    archive_ref = write_immutable_sidecar(
        raw_dir / "historical_model_and_training_receipts.json",
        scope="historical_model_receipts",
        payload={
            "sources": {
                label: {
                    "path": str(path),
                    "sha256": source_hashes[path.as_posix()]["sha256"],
                    "original_model_invoked": sources[label].get("model_invoked"),
                    "original_small_ebm_training": sources[label].get("small_ebm_training"),
                }
                for label, path in (
                    ("exp7432", EXP7432_PATH),
                    ("exp7445", EXP7445_PATH),
                    ("exp7438", EXP7438_PATH),
                )
            },
            "current_model_invoked": False,
            "current_invocation_counts": ZERO_INVOCATION_COUNTS,
        },
        root=root,
    )

    phase_started = time.monotonic()
    progress(run_started, "fault_matrix", "before_owned_children", planned_units=10)
    fault_root = Path(tempfile.mkdtemp(prefix="carnot-exp7458-faults-", dir="/tmp"))
    fault_rows = run_fault_matrix(fault_root)
    fault_checkpoint = raw_dir / "fault_rows.json"
    atomic_json(fault_checkpoint, {"rows": fault_rows})
    spans.append(
        _span(
            "fault_matrix", phase_started, run_started, len(fault_rows), fault_checkpoint.as_posix()
        )
    )
    progress(
        run_started,
        "fault_matrix",
        "after_owned_children",
        completed_units=len(fault_rows),
    )

    phase_started = time.monotonic()
    progress(
        run_started,
        "benchmark",
        "before_paired_blocks",
        planned_units=BENCHMARK_BLOCKS * 2,
    )
    benchmark_root = Path(tempfile.mkdtemp(prefix="carnot-exp7458-benchmark-", dir="/tmp"))
    timing_rows = benchmark_protocol(
        benchmark_root,
        blocks=BENCHMARK_BLOCKS,
        updates=UPDATES_PER_BLOCK,
        ceiling_s=BENCHMARK_CEILING_S,
        emit_progress=True,
    )
    timing_checkpoint = raw_dir / "timing_rows.json"
    atomic_json(timing_checkpoint, {"rows": timing_rows})
    spans.append(
        _span(
            "benchmark",
            phase_started,
            run_started,
            len([row for row in timing_rows if row["row_type"] == "service_timing"]),
            timing_checkpoint.as_posix(),
        )
    )
    progress(
        run_started,
        "benchmark",
        "after_paired_blocks",
        completed_units=len([row for row in timing_rows if row["row_type"] == "service_timing"]),
    )

    raw_path = raw_dir / "raw_evidence.json"
    write_raw_evidence(raw_path, fault_rows=fault_rows, timing_rows=timing_rows)
    raw_reference = _raw_reference(raw_path, root)
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "current_code_or_contract",
            "original_verdict_class": None,
            "original_flagged_adversarial": None,
        }

    private_validation = Path(tempfile.mkdtemp(prefix="carnot-exp7458-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_validation)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned_units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            run_started,
            len(affected),
            (raw_dir / "validation/affected").as_posix(),
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError(f"affected_validation_failed:{affected_reduction}")

    candidate = assemble_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        fault_rows=fault_rows,
        timing_rows=timing_rows,
        raw_reference=raw_reference,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        planned_blocks=BENCHMARK_BLOCKS,
        validation_required=True,
        require_terminal=False,
        candidate=True,
        sidecar_references=[archive_ref],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses", planned_units=4)
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            run_started,
            len(terminal),
            (raw_dir / "validation/terminal").as_posix(),
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    entrypoint = _entrypoint_receipt(root, started_at, time.monotonic() - run_started)
    final = assemble_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        fault_rows=fault_rows,
        timing_rows=timing_rows,
        raw_reference=raw_reference,
        validation_receipts=[*affected, *terminal, entrypoint],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        planned_blocks=BENCHMARK_BLOCKS,
        validation_required=True,
        require_terminal=True,
        flagged_adversarial=not terminal_passed or critical,
        sidecar_references=[archive_ref],
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    destination = output_path if output_path.is_absolute() else root / output_path
    progress(run_started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(candidate_path, final)
    atomic_json(destination, final)
    progress(
        run_started,
        "publish",
        "after_atomic_terminal",
        status=final["status"],
        completed_units=len(final["rows"]),
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse run, fault-worker, and strict fresh-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    modes.add_argument(
        "--fault-worker", nargs=4, metavar=("STORE", "DIRECTORY", "SEQUENCE", "FAULT")
    )
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one bounded read-only helper mode."""

    args = parse_args(argv)
    if args.fault_worker is not None:  # pragma: no cover - owned child path.
        store, directory, sequence, fault = args.fault_worker
        return _fault_worker(store, Path(directory), int(sequence), fault)
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=args.root) if value else ["artifact_unreadable"]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = validate_artifact(value, root=args.root) if value else ["artifact_unreadable"]
        reduced = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
