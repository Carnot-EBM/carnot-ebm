"""Validate a persistent native FIFO archive controller.

The experiment preserves the V637 archive policy. It moves the complete active
and archive state into one Rust object and uses typed calls between snapshots.
It measures semantic readiness only. Exp7257 owns the next throughput study.

Spec refs: REQ-CL-7256, SCENARIO-CL-7256-*, REQ-RUSTPY-7256, and
SCENARIO-RUSTPY-7256-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import ModuleType
from typing import Any
from unittest.mock import patch
import zipfile

import numpy as np

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7243_v637_native_memory as exp7243
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7256
SCHEMA = "carnot.exp7256.v638_native_controller.v1"
MILESTONE = "2026.09.638"
RUN_DATE = "20260913"
RANDOM_SEED = 7_256_000
STREAM_SEEDS = exp7240.STREAM_SEEDS[:8]
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7256_v638_native_controller.json")
NEGATIVE_RELATIVE = Path("results/checkpoints/experiment_7256_v638_negative_fixtures.json")
VALIDATION_RELATIVE = Path("results/checkpoints/experiment_7256_v638_validation_receipts.json")
UPSTREAM_RELATIVE = Path("results/experiment_7243_v637_native_memory.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
CL_SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
RUST_SPEC_RELATIVE = Path("openspec/capabilities/rust-python-boundary/spec.md")
TARGET_RELATIVE = Path("target/experiment-7256-interpreter-bound")
LOAD_RELATIVE = Path("target/experiment-7256-load")
WHEEL_RELATIVE = Path("target/experiment-7256-wheel")
EXPECTED_EXP7243_SHA256 = "sha256:afb3c70b90d73d184fe00f25f776fe5b28bbd5de1eb4f0a472e2aa3a0ed10926"

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("crates/carnot-python/src/packed_belief.rs"),
    Path("crates/carnot-python/src/experiment_7256_archive_controller.rs"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7243_v637_native_memory.py"),
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("scripts/experiments/experiment_7256_v638_native_controller.py"),
    Path("tests/python/test_experiment_7256_v638_native_controller.py"),
    CL_SPEC_RELATIVE,
    RUST_SPEC_RELATIVE,
    UPSTREAM_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "experiment_id": "Bind this result to the fixed experiment identity.",
    "milestone": "Bind this result to milestone 2026.09.638.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260913 and retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start separately from the fixed execution date.",
    "completed_at_utc": "Record the actual UTC end separately from the fixed execution date.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name their device.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Use the closed verdict class; oracle evidence forbids positive classification.",
    "validation_receipts": "Record actual command, exit code and log hash; no test may be skipped or weakened.",
    "baseline_validation_failures": "Record unrelated broad-suite failures separately from scoped acceptance receipts.",
    "native_controller_ready_score": "One requires full semantic parity and persistent native state ownership.",
    "native_binary_receipt": "Bind the interpreter, import path, compiler, wheel and isolated binary to measurements.",
    "parity_rows": "Retain each event and arm, expected and actual decisions, lineage and semantic state.",
    "conversion_count_rows": "Show hot-path conversion and reconstruction counts instead of assuming persistence helps.",
    "throughput_value_score": "Leave throughput unscored because Exp7257 owns the cost experiment.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional evidence apart from the terminal deliverable."""

    checkpoint: Path
    negative_sidecar: Path
    validation_sidecar: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the fixed paths used by the public command."""

        return cls(
            REPO_ROOT / CHECKPOINT_RELATIVE,
            REPO_ROOT / NEGATIVE_RELATIVE,
            REPO_ROOT / VALIDATION_RELATIVE,
            REPO_ROOT / RESULT_RELATIVE,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place test artifacts below one caller-owned temporary directory."""

        return cls(
            root / "checkpoints/parity.json",
            root / "checkpoints/negative.json",
            root / "checkpoints/validation.json",
            root / "experiment_7256.json",
        )


canonical_json = exp7243.canonical_json
sha256_file = exp7243.sha256_file
artifact_checksum = exp7243.artifact_checksum
check = exp7243.check
gate_summary = exp7243.gate_summary
atomic_write = exp7243.atomic_write
_finish_row = exp7243._finish_row


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush a truthful phase boundary for external liveness monitoring."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


class PersistentNativeArchiveController:
    """Expose the persistent Rust controller through the V637 Python API."""

    def __init__(self, binding: ModuleType, *, archive_cap: int = 4) -> None:
        self._binding = binding
        self._native = binding.RustArchiveController7256(archive_cap)

    @classmethod
    def from_snapshot(cls, binding: ModuleType, snapshot: str) -> PersistentNativeArchiveController:
        """Restore only one fully validated explicit native snapshot."""

        controller = cls(binding)
        controller._native.load_snapshot(snapshot)
        return controller

    @classmethod
    def from_state_with_binding(
        cls, binding: ModuleType, state: Mapping[str, Any]
    ) -> PersistentNativeArchiveController:
        """Restore detached mapping bytes at an explicit checkpoint boundary."""

        return cls.from_snapshot(binding, canonical_json(state))

    @classmethod
    def load(cls, binding: ModuleType, path: Path) -> PersistentNativeArchiveController:
        """Load one durable snapshot without changing installed extension state."""

        return cls.from_snapshot(binding, path.read_text(encoding="utf-8"))

    def state_dict(self) -> JsonDict:
        """Return detached state only when the caller requests a snapshot."""

        value = json.loads(self._native.snapshot_state())
        if not isinstance(value, dict):
            raise ValueError("invalid_archive_state_object")
        return value

    def state_bytes(self) -> bytes:
        """Return the same canonical bytes used by the Python reference."""

        return transactional.canonical_json_bytes(self.state_dict())

    def state_hash(self) -> str:
        """Hash native-owned state without reconstructing a Python active object."""

        return str(self._native.state_hash())

    def archives(self) -> list[JsonDict]:
        """Expose immutable archive receipts at an explicit audit boundary."""

        return deepcopy(self.state_dict()["archives"])

    def conversion_counts(self) -> JsonDict:
        """Return native counters that expose all restore and typed-call boundaries."""

        return dict(self._native.conversion_counts())

    def save(self, path: Path) -> JsonDict:
        """Atomically save one complete native snapshot."""

        return transactional._atomic_write(path, self.state_bytes())

    def predict(self, event: Mapping[str, Any]) -> tuple[str, float]:
        """Send one public event as typed family and value fields."""

        coordinates = exp7226.PackedBeliefController._public_coordinates(event)
        if coordinates is None:
            return "abstain", 0.0
        family, value = coordinates
        decision, disagreement = self._native.predict(exp7230.FAMILY_CODES[family], value)
        return str(decision), float(disagreement)

    def energy(self, label: str, event: Mapping[str, Any]) -> JsonDict:
        """Send one typed energy query and preserve public error semantics."""

        if label not in {"accept", "reject"}:
            return {
                "status": "unknown_label",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        coordinates = exp7226.PackedBeliefController._public_coordinates(event)
        if coordinates is None:
            return {
                "status": "unknown_input",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        family, value = coordinates
        return dict(
            self._native.energy(int(label == "accept"), exp7230.FAMILY_CODES[family], value)
        )

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Mapping[str, int],
    ) -> Mapping[str, Any]:
        """Select by disagreement and stable rank inside the same native object."""

        coordinates = [exp7226.PackedBeliefController._public_coordinates(row) for row in block]
        if not block or any(value is None for value in coordinates):
            return exp7240.exp7199.select_request(block, "priority_admission", tie_ranks, self)
        typed = [value for value in coordinates if value is not None]
        index = int(
            self._native.select_request(
                np.ascontiguousarray(
                    [exp7230.FAMILY_CODES[family] for family, _ in typed], dtype=np.uint8
                ),
                np.ascontiguousarray([value for _, value in typed], dtype=np.int64),
                np.ascontiguousarray(
                    [tie_ranks[str(row["event_id"])] for row in block], dtype=np.int64
                ),
            )
        )
        return block[index]

    def _durable_parent_matches(self, path: Path, expected_hash: str) -> None:
        """Reject stale or malformed durable bytes before native mutation."""

        if not path.exists():
            return
        try:
            observed = type(self).load(self._binding, path).state_hash()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise exp7240.ArchiveCommitRejected("corrupt_durable_state") from error
        if observed != expected_hash:
            raise exp7240.ArchiveCommitRejected("stale_durable_parent")

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Apply typed releases and undo native admission if publication fails."""

        if state_path is not None:
            self._durable_parent_matches(state_path, expected_parent_hash)
        try:
            normalized = [
                exp7226.PackedBeliefController._validate_release(row, current_cycle)
                for row in releases
            ]
        except exp7226.CommitRejected as error:
            raise exp7240.ArchiveCommitRejected(str(error)) from error
        try:
            raw = self._native.commit_batch(
                [str(row["event_id"]) for row in normalized],
                np.ascontiguousarray(
                    [exp7230.FAMILY_CODES[str(row["family_id"])] for row in normalized],
                    dtype=np.uint8,
                ),
                np.ascontiguousarray([row["numeric_value"] for row in normalized], dtype=np.int64),
                np.ascontiguousarray(
                    [int(row["observed_label"] == "accept") for row in normalized],
                    dtype=np.int8,
                ),
                np.ascontiguousarray(
                    [int(row["role"] == "support") for row in normalized], dtype=np.uint8
                ),
                np.ascontiguousarray([row["request_index"] for row in normalized], dtype=np.int64),
                np.ascontiguousarray([row["release_index"] for row in normalized], dtype=np.int64),
                current_cycle,
                expected_parent_hash,
            )
        except ValueError as error:
            raise exp7240.ArchiveCommitRejected(str(error)) from error
        native_receipt = json.loads(raw)
        parent_bytes = str(native_receipt.pop("parent_json")).encode()
        new_bytes = str(native_receipt.pop("new_json")).encode()
        receipt = {
            **native_receipt,
            "parent_bytes_b64": transactional.encode_bytes(parent_bytes),
            "new_state_bytes_b64": transactional.encode_bytes(new_bytes),
            "atomic_write": None,
        }
        if state_path is not None:
            try:
                receipt["atomic_write"] = transactional._atomic_write(state_path, new_bytes)
            except OSError:
                self._native.rollback(receipt["new_state_hash"], receipt["parent_hash"])
                raise
        return receipt

    def rollback(self, receipt: Mapping[str, Any], *, state_path: Path | None = None) -> JsonDict:
        """Restore the retained native parent only from its exact child."""

        if self.state_hash() != receipt.get("new_state_hash"):
            raise exp7240.ArchiveCommitRejected("stale_rollback")
        try:
            parent_bytes = transactional.decode_bytes(str(receipt["parent_bytes_b64"]))
            parent = type(self).from_snapshot(self._binding, parent_bytes.decode())
        except (KeyError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise exp7240.ArchiveCommitRejected("invalid_rollback_receipt") from error
        if parent.state_hash() != receipt.get("parent_hash"):
            raise exp7240.ArchiveCommitRejected("rollback_parent_hash")
        if state_path is not None:
            self._durable_parent_matches(state_path, str(receipt["new_state_hash"]))
            transactional._atomic_write(state_path, parent_bytes)
        try:
            self._native.rollback(str(receipt["new_state_hash"]), str(receipt["parent_hash"]))
        except ValueError as error:
            raise exp7240.ArchiveCommitRejected(str(error)) from error
        return {
            "parent_hash": receipt["parent_hash"],
            "restored_state_hash": self.state_hash(),
            "byte_identical": self.state_bytes() == parent_bytes,
            "atomic_write": None,
        }


class _CountingPythonArchive(exp7240.ArchivedBeliefController):
    """Count actual active reconstructions in the shipped Python reference."""

    def __init__(self, **kwargs: Any) -> None:
        self.active_reconstruction_count = 0
        super().__init__(**kwargs)

    def _active_from_state(self, value: Mapping[str, Any]) -> Any:
        self.active_reconstruction_count = getattr(self, "active_reconstruction_count", 0) + 1
        return super()._active_from_state(value)


class _CountingOldNativeArchive(exp7243.NativeArchiveController):
    """Count active reconstructions at the former Python/native boundary."""

    def __init__(self, binding: ModuleType, **kwargs: Any) -> None:
        self.active_reconstruction_count = 0
        super().__init__(binding, **kwargs)

    def _active_from_state(self, value: Mapping[str, Any]) -> Any:
        self.active_reconstruction_count = getattr(self, "active_reconstruction_count", 0) + 1
        return super()._active_from_state(value)


def _build_wheel(extension: Path, wheel_dir: Path) -> Path:
    """Package the selected extension in an isolated wheel evidence path."""

    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = wheel_dir / "carnot_native_7256-0.1.0-cp312-cp312-linux_x86_64.whl"
    metadata = "Metadata-Version: 2.1\nName: carnot-native-7256\nVersion: 0.1.0\n"
    wheel_metadata = "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(extension, f"carnot/{extension.name}")
        archive.writestr("carnot_native_7256-0.1.0.dist-info/METADATA", metadata)
        archive.writestr("carnot_native_7256-0.1.0.dist-info/WHEEL", wheel_metadata)
        archive.writestr("carnot_native_7256-0.1.0.dist-info/RECORD", "")
    return wheel


def build_native_extension(root: Path, *, show_progress: bool = True) -> tuple[Path, JsonDict]:
    """Build and package only carnot-python for the running interpreter."""

    target = root / TARGET_RELATIVE
    environment = exp7217.interpreter_build_environment(Path(sys.executable), target)
    if show_progress:
        progress(3, "before", "isolated interpreter-bound native build subprocess")
    receipt = exp7217._stream_process(
        ["cargo", "build", "--release", "-p", "carnot-python"],
        root=root,
        environment=environment,
        operation="Exp7256 interpreter-bound carnot-python build",
        timeout_s=900,
    )
    library = target / "release/libcarnot_python.so"
    if not library.is_file():
        raise RuntimeError(f"native build output missing: {library}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix unavailable")
    load_dir = root / LOAD_RELATIVE
    load_dir.mkdir(parents=True, exist_ok=True)
    destination = load_dir / f"_rust{suffix}"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=load_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, destination)
        destination.chmod(0o755)
    finally:
        if temporary.exists():
            temporary.unlink()
    wheel = _build_wheel(destination, root / WHEEL_RELATIVE)
    compiler = subprocess.run(
        ["rustc", "--version", "--verbose"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    receipt.update(
        {
            "PYO3_PYTHON": environment["PYO3_PYTHON"],
            "CARGO_TARGET_DIR": environment["CARGO_TARGET_DIR"],
            "built_library": str(library.resolve()),
            "built_library_sha256": sha256_file(library),
            "loaded_copy": str(destination.resolve()),
            "module_sha256": sha256_file(destination),
            "wheel_path": str(wheel.resolve()),
            "wheel_sha256": sha256_file(wheel),
            "compiler_command": ["rustc", "--version", "--verbose"],
            "compiler_exit_code": compiler.returncode,
            "compiler_output": compiler.stdout,
        }
    )
    if show_progress:
        progress(3, "after", f"native build selected={destination.resolve()}")
    return destination, receipt


def collect_preconditions(root: Path, paths: ExperimentPaths) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate upstream bytes, quarantine, imports, tools, and output ownership."""

    progress(0, "start", "authenticate upstream bytes, quarantine, imports, and output paths")
    hashes = {
        str(path): (sha256_file(root / path) if (root / path).is_file() else None)
        for path in SOURCE_PATHS
    }
    upstream = exp7243.read_object(root / UPSTREAM_RELATIVE)
    exclusion = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    quarantine = exp7213.quarantine_state(
        upstream, exclusion, UPSTREAM_RELATIVE.name, "exp7243-native-memory"
    )
    cl_text = (root / CL_SPEC_RELATIVE).read_text(encoding="utf-8")
    rust_text = (root / RUST_SPEC_RELATIVE).read_text(encoding="utf-8")
    tools = {name: shutil.which(name) for name in ("cargo", "rustc", "ldd")}
    outputs = {
        "checkpoint": _writable(paths.checkpoint),
        "negative_sidecar": _writable(paths.negative_sidecar),
        "validation_sidecar": _writable(paths.validation_sidecar),
        "artifact": _writable(paths.artifact),
    }
    try:
        checksum_valid = exp7243.artifact_checksum(upstream) == upstream.get(
            "reproducibility_checksum"
        )
    except (TypeError, ValueError):
        checksum_valid = False
    checks = [
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            {name: value for name, value in hashes.items()},
            all(value is not None for value in hashes.values()),
        ),
        check(
            "driving_capability_specs",
            f"{CL_SPEC_RELATIVE},{RUST_SPEC_RELATIVE}",
            "REQ-CL-7256,REQ-RUSTPY-7256",
            {"REQ-CL-7256": True, "REQ-RUSTPY-7256": True},
            {
                "REQ-CL-7256": "REQ-CL-7256" in cl_text,
                "REQ-RUSTPY-7256": "REQ-RUSTPY-7256" in rust_text,
            },
            "REQ-CL-7256" in cl_text and "REQ-RUSTPY-7256" in rust_text,
        ),
        check(
            "exp7243_artifact_hash",
            str(UPSTREAM_RELATIVE),
            "sha256",
            EXPECTED_EXP7243_SHA256,
            hashes[str(UPSTREAM_RELATIVE)],
            hashes[str(UPSTREAM_RELATIVE)] == EXPECTED_EXP7243_SHA256,
        ),
        check(
            "exp7243_not_quarantined",
            str(UPSTREAM_RELATIVE),
            "quarantined",
            False,
            quarantine,
            quarantine.get("quarantined") is False,
        ),
        check(
            "exp7243_ready_and_complete",
            str(UPSTREAM_RELATIVE),
            "status,native_archive_ready_score,reproducibility_checksum",
            {"status": "complete", "ready": 1, "checksum": True},
            {
                "status": upstream.get("status"),
                "ready": upstream.get("native_archive_ready_score"),
                "checksum": checksum_valid,
            },
            upstream.get("status") == "complete"
            and upstream.get("native_archive_ready_score") == 1
            and checksum_valid,
        ),
        check(
            "imports_tools_and_outputs",
            "host",
            "imports,compiler,linker,owned paths",
            "all available and writable",
            {"tools": tools, "outputs": outputs, "numpy": np.__version__},
            all(tools.values())
            and all(outputs.values())
            and hasattr(exp7243, "NativeArchiveController"),
        ),
    ]
    progress(
        0,
        "end",
        f"precondition checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
    )
    return checks, {"hashes": hashes, "upstream": upstream, "quarantine": quarantine}


def _event_row(
    event: Mapping[str, Any],
    arm: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> JsonDict:
    """Retain one event and arm with decisions, lineage, and semantic state."""

    mismatch = int(expected != actual)
    return {
        "event_id": event["event_id"],
        "chronology_index": event["chronology_index"],
        "arm": arm,
        "expected": deepcopy(dict(expected)),
        "actual": deepcopy(dict(actual)),
        "mismatch_count": mismatch,
    }


def _controller_observation(
    controller: Any,
    event: Mapping[str, Any],
    selected_id: str,
) -> JsonDict:
    """Read the complete decision-bearing state before released feedback."""

    return {
        "prediction": controller.predict(event),
        "energies": [controller.energy(label, event) for label in ("accept", "reject")],
        "query_selected": str(event["event_id"]) == selected_id,
        "state_hash": controller.state_hash(),
        "archive_ids": [row["archive_id"] for row in controller.archives()],
        "release_count": len(controller.state_dict()["release_ids"]),
    }


def run_differential_replay(
    binding: ModuleType,
    *,
    stream_count: int = 8,
    event_limit: int = exp7240.EVENTS_PER_STREAM,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay eight fixed streams through all three controller boundaries."""

    public, authority, schedules = exp7243._stream_sources(REPO_ROOT)
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    loop_started = time.monotonic()
    last_report = loop_started
    for stream_offset, seed in enumerate(STREAM_SEEDS[:stream_count]):
        stream_id = f"stream-{stream_offset + 1:02d}"
        events = [row for row in public if row["stream_id"] == stream_id][:event_limit]
        reference = _CountingPythonArchive(archive_cap=4, nomination_mode="validated")
        old = _CountingOldNativeArchive(binding, archive_cap=4)
        persistent = PersistentNativeArchiveController(binding, archive_cap=4)
        controllers = {
            "python_reference": reference,
            "old_native_wrapper": old,
            "persistent_native_controller": persistent,
        }
        event_rows = {arm: [] for arm in controllers}
        mismatch_counts = dict.fromkeys(controllers, 0)
        maximum_archives = dict.fromkeys(controllers, 0)
        pending: list[JsonDict] = []
        query_count = 0
        contradiction_count = 0
        for block_index, offset in enumerate(range(0, len(events), exp7240.QUERY_BLOCK_SIZE)):
            block = events[offset : offset + exp7240.QUERY_BLOCK_SIZE]
            tie_ranks = exp7240.exp7199.seeded_tie_ranks(seed, block_index, block)
            selected = {
                arm: str(controller.select_request(block, tie_ranks)["event_id"])
                for arm, controller in controllers.items()
            }
            expected_selected = selected["python_reference"]
            for event in block:
                event_id = str(event["event_id"])
                chronology = int(event["chronology_index"])
                expected = _controller_observation(reference, event, selected["python_reference"])
                for arm, controller in controllers.items():
                    actual = _controller_observation(controller, event, selected[arm])
                    row = _event_row(event, arm, expected, actual)
                    mismatch_counts[arm] += row["mismatch_count"]
                    event_rows[arm].append(row)
                    maximum_archives[arm] = max(maximum_archives[arm], len(actual["archive_ids"]))
                will_query = (
                    event_id == expected_selected
                    and query_count < exp7240.QUERY_CEILING
                    and len(pending) < exp7240.PENDING_CAPACITY
                )
                if will_query:
                    query_count += 1
                    pending.append(
                        {
                            "public": deepcopy(event),
                            "observed_label": authority[event_id]["exact_label"],
                            "request_index": chronology,
                            "release_index": chronology + int(schedules[event_id]["delay"]),
                        }
                    )
                due = sorted(
                    [row for row in pending if row["release_index"] <= chronology],
                    key=lambda row: (row["release_index"], row["request_index"]),
                )
                if due:
                    releases = [exp7240._support_release(row) for row in due]
                    receipts = {}
                    for arm, controller in controllers.items():
                        receipts[arm] = controller.commit_batch(
                            releases,
                            current_cycle=chronology,
                            expected_parent_hash=controller.state_hash(),
                        )
                    expected_receipt = receipts["python_reference"]
                    contradiction_count += sum(
                        int(operation["active_contradiction"])
                        for operation in expected_receipt["operations"]
                    )
                    for arm, controller in controllers.items():
                        mismatch_counts[arm] += int(
                            receipts[arm]["operations"] != expected_receipt["operations"]
                            or controller.state_bytes() != reference.state_bytes()
                        )
                    pending = [row for row in pending if row not in due]
                now = time.monotonic()
                if now - last_report >= 60:
                    completed = stream_offset * event_limit + min(offset + len(block), len(events))
                    print(
                        f"[phase 4 progress] completed_events={completed}/{stream_count * event_limit} "
                        f"elapsed_s={now - loop_started:.3f}",
                        flush=True,
                    )
                    last_report = now

        rollback_parent = reference.state_bytes()
        probe = {
            "event_id": f"rollback-{stream_id}",
            "family_id": exp7240.FAMILIES[stream_offset % 4],
            "numeric_value": stream_offset * 5,
        }
        prediction = reference.predict(probe)[0]
        release = {
            **probe,
            "observed_label": "reject" if prediction == "accept" else "accept",
            "role": "support",
            "request_index": 4_000,
            "release_index": 4_000,
        }
        receipts = {
            arm: controller.commit_batch(
                [release], current_cycle=4_000, expected_parent_hash=controller.state_hash()
            )
            for arm, controller in controllers.items()
        }
        for arm, controller in controllers.items():
            controller.rollback(receipts[arm])
            mismatch_counts[arm] += int(controller.state_bytes() != rollback_parent)

        for arm, controller in controllers.items():
            if arm == "python_reference":
                counts = {
                    "active_reconstruction_count": reference.active_reconstruction_count,
                    "hot_path_json_parse_count": 0,
                    "snapshot_json_parse_count": 0,
                    "typed_event_calls": 0,
                }
            elif arm == "old_native_wrapper":
                counts = {
                    "active_reconstruction_count": old.active_reconstruction_count,
                    "hot_path_json_parse_count": old.active_reconstruction_count,
                    "snapshot_json_parse_count": old.active_reconstruction_count,
                    "typed_event_calls": old.native_call_count,
                }
            else:
                counts = persistent.conversion_counts()
            rows.append(
                _finish_row(
                    {
                        "unit_id": f"parity:{stream_id}:{arm}",
                        "stream_id": stream_id,
                        "arm": arm,
                        "seed": seed,
                        "metric": mismatch_counts[arm],
                        "error": None,
                        "abstention": sum(
                            int(row["actual"]["prediction"][0] == "abstain")
                            for row in event_rows[arm]
                        ),
                        "censored": False,
                        "completed_events": len(events),
                        "event_rows": event_rows[arm],
                        "mismatch_count": mismatch_counts[arm],
                        "maximum_archive_slots": maximum_archives[arm],
                        "archive_slots_configured": 4,
                        "conflicting_delayed_release_count": contradiction_count,
                        "rollback_byte_identical": controller.state_bytes() == rollback_parent,
                        "final_state_hash": controller.state_hash(),
                        "conversion_counts": counts,
                        "passed": mismatch_counts[arm] == 0,
                    }
                )
            )
        checkpoints.append(exp7243._continuation_fixture(reference.state_dict(), seed))
        print(
            f"[phase 4 progress] completed_streams={stream_offset + 1}/{stream_count} "
            f"completed_events={(stream_offset + 1) * len(events)}/{stream_count * event_limit} "
            f"elapsed_s={time.monotonic() - loop_started:.3f}",
            flush=True,
        )
    return rows, checkpoints


def _continue_controller(controller: Any, checkpoint: Mapping[str, Any]) -> JsonDict:
    """Continue fixed typed probes and delayed releases from one snapshot."""

    return exp7243._continue_controller(controller, checkpoint)


_CONTINUATION_HELPER = r"""
import importlib.util
import json
from pathlib import Path
import sys
from carnot import experiment_7256_v638_native_controller as experiment

extension = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("carnot._rust", extension)
if spec is None or spec.loader is None:
    raise RuntimeError("native loader unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
checkpoint = json.loads(sys.stdin.read())
controller = experiment.PersistentNativeArchiveController.from_state_with_binding(
    module, checkpoint["state"]
)
result = experiment._continue_controller(controller, checkpoint)
result["module_file"] = str(Path(module.__file__).resolve())
print("__CARNOT_JSON__" + json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
"""


def run_fresh_process_continuation(extension: Path, checkpoint: Mapping[str, Any]) -> JsonDict:
    """Restore exact native bytes in a fresh process and compare next decisions."""

    progress(5, "before", "fresh-process persistent-native restore and continuation")
    expected = json.loads(
        canonical_json(
            _continue_controller(
                exp7240.ArchivedBeliefController.from_state(checkpoint["state"]), checkpoint
            )
        )
    )
    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _CONTINUATION_HELPER,
        str(extension),
    ]
    started = time.monotonic()
    with exp7217._Heartbeat("Exp7256 fresh-process continuation", interval_s=30):
        completed = subprocess.run(
            command,
            input=canonical_json(checkpoint),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT / "python")},
        )
    observed: JsonDict = {}
    for line in completed.stdout.splitlines():
        if line.startswith("__CARNOT_JSON__"):
            value = json.loads(line.removeprefix("__CARNOT_JSON__"))
            if isinstance(value, Mapping):
                observed = dict(value)
    same = (
        completed.returncode == 0
        and observed.get("outputs") == expected["outputs"]
        and observed.get("final_state_hash") == expected["final_state_hash"]
        and observed.get("final_state") == expected["final_state"]
    )
    receipt = _finish_row(
        {
            "unit_id": "fresh-process-next-decisions",
            "arm": "persistent_native_fresh_process",
            "seed": STREAM_SEEDS[-1],
            "metric": int(not same),
            "error": None if completed.returncode == 0 else completed.stderr,
            "abstention": False,
            "censored": False,
            "command": command,
            "exit_code": completed.returncode,
            "duration_s": time.monotonic() - started,
            "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": "sha256:" + hashlib.sha256(completed.stderr.encode()).hexdigest(),
            "module_file": observed.get("module_file"),
            "same_next_decisions": same,
            "same_semantic_state": observed.get("final_state") == expected["final_state"],
            "mismatch_count": int(not same),
            "passed": same and observed.get("module_file") == str(extension.resolve()),
        }
    )
    progress(5, "after", f"fresh-process continuation passed={receipt['passed']}")
    return receipt


def run_negative_controls(binding: ModuleType, durable: Path) -> list[JsonDict]:
    """Attack snapshot, parent, and durable writes without changing live bytes."""

    controls = []
    controller = PersistentNativeArchiveController(binding)
    parent = controller.state_bytes()
    malformed_rejected = False
    try:
        PersistentNativeArchiveController.from_snapshot(binding, "{}")
    except ValueError:
        malformed_rejected = True
    controls.append(
        {"control": "malformed_snapshot", "passed": malformed_rejected, "parent_preserved": True}
    )
    stale_rejected = False
    release = {
        "event_id": "negative-stale",
        "family_id": "lower_bound",
        "numeric_value": 1,
        "observed_label": "accept",
        "role": "support",
        "request_index": 0,
        "release_index": 0,
    }
    try:
        controller.commit_batch([release], current_cycle=0, expected_parent_hash="sha256:bad")
    except exp7240.ArchiveCommitRejected:
        stale_rejected = True
    controls.append(
        {
            "control": "stale_parent",
            "passed": stale_rejected and controller.state_bytes() == parent,
            "parent_preserved": controller.state_bytes() == parent,
        }
    )
    controller.save(durable)
    interrupted_rejected = False
    with patch.object(transactional, "_atomic_write", side_effect=OSError("injected interrupt")):
        try:
            controller.commit_batch(
                [release],
                current_cycle=0,
                expected_parent_hash=controller.state_hash(),
                state_path=durable,
            )
        except OSError:
            interrupted_rejected = True
    controls.append(
        {
            "control": "interrupted_write",
            "passed": interrupted_rejected
            and controller.state_bytes() == parent
            and durable.read_bytes() == parent,
            "parent_preserved": controller.state_bytes() == parent,
            "durable_parent_preserved": durable.read_bytes() == parent,
        }
    )
    return [
        _finish_row(
            {
                "unit_id": f"negative:{row['control']}",
                "arm": "persistent_native_controller",
                "seed": RANDOM_SEED,
                "metric": int(not row["passed"]),
                "error": None,
                "abstention": False,
                "censored": False,
                **row,
            }
        )
        for row in controls
    ]


def reduce_parity_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce nested event rows without trusting producer totals."""

    event_mismatches = sum(
        int(event["expected"] != event["actual"])
        for row in rows
        for event in row.get("event_rows", [])
    )
    declared_mismatches = sum(int(row.get("mismatch_count", 0)) for row in rows)
    return {
        "stream_arm_rows": len(rows),
        "event_arm_rows": sum(len(row.get("event_rows", [])) for row in rows),
        "event_mismatch_count": event_mismatches,
        "declared_mismatch_count": declared_mismatches,
        "all_rollbacks_exact": all(row.get("rollback_byte_identical") is True for row in rows),
        "all_rows_passed": all(row.get("passed") is True for row in rows),
    }


def _conversion_rows(parity_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate measured boundary counts for each controller arm."""

    rows = []
    for arm in ("python_reference", "old_native_wrapper", "persistent_native_controller"):
        selected = [row for row in parity_rows if row.get("arm") == arm]
        counts = {
            name: sum(int(row["conversion_counts"].get(name, 0)) for row in selected)
            for name in (
                "typed_event_calls",
                "hot_path_json_parse_count",
                "snapshot_json_parse_count",
                "active_reconstruction_count",
            )
        }
        rows.append(
            _finish_row(
                {
                    "unit_id": f"conversion:{arm}",
                    "arm": arm,
                    "seed": RANDOM_SEED,
                    "metric": counts["active_reconstruction_count"],
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    **counts,
                }
            )
        )
    return rows


def _sample_budget(complete: bool, event_limit: int = exp7240.EVENTS_PER_STREAM) -> JsonDict:
    """Declare the fixed independent units and no-extension stopping rule."""

    return {
        "independent_units_planned": 8,
        "independent_units_attempted": 8 if complete else 0,
        "independent_units_completed": 8 if complete else 0,
        "independent_units_censored": 0 if complete else 8,
        "controller_arms": 3,
        "events_per_stream": event_limit,
        "event_arm_rows_planned": 8 * event_limit * 3,
        "stopping_rule": "eight fixed existing streams once; no outcome extension",
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
    """Create all required fields before terminal classification."""

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
        "current_model_count": 0,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": {"global": RANDOM_SEED, "streams": list(STREAM_SEEDS)},
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(False),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external:unknown_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "baseline_validation_failures": [],
        "native_controller_ready_score": 0,
        "native_binary_receipt": {},
        "parity_rows": [],
        "conversion_count_rows": [],
        "throughput_value_score": None,
        "fresh_process_receipt": {},
        "negative_control_rows": [],
        "independent_reducer_receipt": {},
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None},
        "negative_fixture_receipt": {"path": str(paths.negative_sidecar), "sha256": None},
        "upstream_receipt": {
            "path": str(UPSTREAM_RELATIVE),
            "sha256": hashes.get(str(UPSTREAM_RELATIVE)),
            "historical_models_currently_invoked": False,
        },
        "default_pipeline_modified": False,
        "publication_performed": False,
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return a schema-complete row-free external-block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["honest_verdict"] = f"blocked_external:{failed.get('check')}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_parity_rows() -> list[JsonDict]:
    """Build a compact complete three-arm parity fixture for cold tests."""

    rows = []
    for stream_index, seed in enumerate(STREAM_SEEDS):
        for arm in ("python_reference", "old_native_wrapper", "persistent_native_controller"):
            observation = {
                "prediction": ["reject", 0.0],
                "energies": [],
                "query_selected": True,
                "state_hash": "sha256:fixture",
                "archive_ids": [],
                "release_count": 0,
            }
            counts = {
                "typed_event_calls": int(arm != "python_reference"),
                "hot_path_json_parse_count": int(arm == "old_native_wrapper"),
                "snapshot_json_parse_count": int(arm == "old_native_wrapper"),
                "active_reconstruction_count": int(arm != "persistent_native_controller"),
            }
            rows.append(
                _finish_row(
                    {
                        "unit_id": f"parity:stream-{stream_index + 1:02d}:{arm}",
                        "stream_id": f"stream-{stream_index + 1:02d}",
                        "arm": arm,
                        "seed": seed,
                        "metric": 0,
                        "error": None,
                        "abstention": False,
                        "censored": False,
                        "completed_events": 1,
                        "event_rows": [
                            {
                                "event_id": "fixture",
                                "chronology_index": 0,
                                "arm": arm,
                                "expected": observation,
                                "actual": observation,
                                "mismatch_count": 0,
                            }
                        ],
                        "mismatch_count": 0,
                        "maximum_archive_slots": 4,
                        "archive_slots_configured": 4,
                        "conflicting_delayed_release_count": 1,
                        "rollback_byte_identical": True,
                        "final_state_hash": "sha256:fixture",
                        "conversion_counts": counts,
                        "passed": True,
                    }
                )
            )
    return rows


def complete_artifact_fixture_for_test() -> JsonDict:
    """Create one complete circular-positive structural fixture."""

    now = datetime.now(UTC).isoformat()
    checks = [check("fixture", "fixture", "value", 1, 1, True)]
    artifact = _base_artifact(
        checks, {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    parity = _fixture_parity_rows()
    conversions = _conversion_rows(parity)
    fresh = _finish_row(
        {
            "unit_id": "fresh-process-next-decisions",
            "arm": "persistent_native_fresh_process",
            "seed": STREAM_SEEDS[-1],
            "metric": 0,
            "error": None,
            "abstention": False,
            "censored": False,
            "same_next_decisions": True,
            "same_semantic_state": True,
            "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
            "passed": True,
        }
    )
    negatives = [
        _finish_row(
            {
                "unit_id": f"negative:{name}",
                "arm": "persistent_native_controller",
                "seed": RANDOM_SEED,
                "metric": 0,
                "error": None,
                "abstention": False,
                "censored": False,
                "control": name,
                "parent_preserved": True,
                "passed": True,
            }
        )
        for name in ("malformed_snapshot", "stale_parent", "interrupted_write")
    ]
    reducer = reduce_parity_rows(parity)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": [*parity, *conversions, fresh, *negatives],
            "sample_size_budget": _sample_budget(True, 1),
            "acceptance_gate_results": {
                "exact_three_arm_parity": {
                    "principle": "Every semantic mismatch blocks readiness.",
                    "expected": 0,
                    "observed": 0,
                    "passed": True,
                },
                "persistent_native_ownership": {
                    "principle": "The hot path must not parse state or reconstruct active objects.",
                    "expected": {"hot_path_json_parse_count": 0, "active_reconstruction_count": 0},
                    "observed": {"hot_path_json_parse_count": 0, "active_reconstruction_count": 0},
                    "passed": True,
                },
                "restart_and_transactions": {
                    "principle": "Restore and rejected transactions preserve exact semantics and bytes.",
                    "expected": True,
                    "observed": True,
                    "passed": True,
                },
            },
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive: exact oracle parity and persistent native ownership passed; throughput deferred",
            "validation_receipts": [
                {
                    "name": "fixture",
                    "command": ["fixture"],
                    "exit_code": 0,
                    "log_sha256": "sha256:fixture",
                }
            ],
            "native_controller_ready_score": 1,
            "native_binary_receipt": {
                "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
                "module_sha256": "sha256:fixture",
                "wheel_path": "/tmp/fixture.whl",
                "wheel_sha256": "sha256:fixture",
                "compiler_output": "rustc fixture",
                "native_class": "carnot._rust.RustArchiveController7256",
                "compiled_execution": True,
                "python_fallback_used": False,
            },
            "parity_rows": parity,
            "conversion_count_rows": conversions,
            "fresh_process_receipt": fresh,
            "negative_control_rows": negatives,
            "independent_reducer_receipt": reducer,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, check_files: bool = False, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check provenance, semantic rows, counters, gates, and identity."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS.issubset(artifact), "missing_fields")
    if not REQUIRED_FIELDS.issubset(artifact):
        return errors
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
        or any(
            artifact.get(name) != 0
            for name in (
                "current_model_count",
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "model_invoked",
    )
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    add(not checksum_valid, "reproducibility_checksum")
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "blocked_class")
        add(artifact.get("rows") != [], "blocked_rows")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        return errors

    parity = artifact.get("parity_rows", [])
    conversions = artifact.get("conversion_count_rows", [])
    fresh = artifact.get("fresh_process_receipt", {})
    negatives = artifact.get("negative_control_rows", [])
    add(artifact.get("status") != "complete", "status")
    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(
        artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_class",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    add(
        not isinstance(parity, list)
        or len(parity) != 24
        or {row.get("arm") for row in parity}
        != {"python_reference", "old_native_wrapper", "persistent_native_controller"}
        or any(row.get("mismatch_count") != 0 or row.get("passed") is not True for row in parity),
        "parity_rows",
    )
    try:
        reducer = reduce_parity_rows(parity)
    except (KeyError, TypeError, ValueError):
        reducer = None
    add(
        reducer is None or artifact.get("independent_reducer_receipt") != reducer, "raw_row_reducer"
    )
    persistent = next(
        (row for row in conversions if row.get("arm") == "persistent_native_controller"), {}
    )
    add(
        len(conversions) != 3
        or persistent.get("typed_event_calls", 0) <= 0
        or persistent.get("hot_path_json_parse_count") != 0
        or persistent.get("active_reconstruction_count") != 0,
        "conversion_count_rows",
    )
    add(
        not isinstance(fresh, Mapping)
        or fresh.get("passed") is not True
        or fresh.get("same_next_decisions") is not True
        or fresh.get("same_semantic_state") is not True,
        "fresh_process_receipt",
    )
    add(
        not isinstance(negatives, list)
        or {row.get("control") for row in negatives}
        != {"malformed_snapshot", "stale_parent", "interrupted_write"}
        or any(row.get("passed") is not True for row in negatives),
        "negative_controls",
    )
    identity = artifact.get("native_binary_receipt", {})
    add(
        not isinstance(identity, Mapping)
        or identity.get("compiled_execution") is not True
        or identity.get("python_fallback_used") is not False
        or identity.get("native_class") != "carnot._rust.RustArchiveController7256"
        or not str(identity.get("module_file", "")).endswith(
            str(sysconfig.get_config_var("EXT_SUFFIX"))
        )
        or not str(identity.get("module_sha256", "")).startswith("sha256:")
        or not str(identity.get("wheel_sha256", "")).startswith("sha256:"),
        "native_binary_receipt",
    )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or not receipts
        or any(
            not isinstance(row.get("command"), list)
            or row.get("exit_code") != 0
            or not str(row.get("log_sha256", "")).startswith("sha256:")
            for row in receipts
        ),
        "validation_receipts",
    )
    ready = int(
        reducer is not None
        and reducer["event_mismatch_count"] == 0
        and reducer["declared_mismatch_count"] == 0
        and reducer["all_rollbacks_exact"] is True
        and persistent.get("hot_path_json_parse_count") == 0
        and persistent.get("active_reconstruction_count") == 0
        and fresh.get("passed") is True
        and all(row.get("passed") is True for row in negatives)
        and identity.get("compiled_execution") is True
    )
    add(artifact.get("native_controller_ready_score") != ready, "ready_score")
    add(artifact.get("throughput_value_score") is not None, "throughput_value_score")
    add(artifact.get("verdict_class") != "circular_positive", "verdict_class")
    add(not str(artifact.get("honest_verdict", "")).startswith("complete_"), "honest_verdict")
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or not gates
        or any(
            set(row) != {"principle", "expected", "observed", "passed"} for row in gates.values()
        ),
        "acceptance_gate_results",
    )
    if check_files:
        for path_text, expected_hash in artifact.get("source_artifact_hashes", {}).items():
            path = Path(path_text)
            resolved = path if path.is_absolute() else root / path
            add(
                not resolved.is_file() or sha256_file(resolved) != expected_hash,
                "source_artifact_hashes",
            )
        add(
            not Path(str(identity.get("module_file", ""))).is_file()
            or sha256_file(Path(str(identity["module_file"]))) != identity.get("module_sha256"),
            "native_module_hash",
        )
        add(
            not Path(str(identity.get("wheel_path", ""))).is_file()
            or sha256_file(Path(str(identity["wheel_path"]))) != identity.get("wheel_sha256"),
            "native_wheel_hash",
        )
    return errors


def _native_identity(
    extension: Path,
    binding: ModuleType,
    build_receipt: Mapping[str, Any],
) -> JsonDict:
    """Bind the running interpreter, compiler, wheel, and imported native bytes."""

    controller = binding.RustArchiveController7256()
    methods = [
        name
        for name in (
            "predict",
            "energy",
            "select_request",
            "commit_batch",
            "snapshot_state",
            "load_snapshot",
            "rollback",
            "conversion_counts",
        )
        if hasattr(controller, name)
    ]
    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "module_file": str(Path(binding.__file__).resolve()),
        "module_sha256": sha256_file(extension),
        "built_library": build_receipt["built_library"],
        "built_library_sha256": build_receipt["built_library_sha256"],
        "wheel_path": build_receipt["wheel_path"],
        "wheel_sha256": build_receipt["wheel_sha256"],
        "compiler_command": build_receipt["compiler_command"],
        "compiler_exit_code": build_receipt["compiler_exit_code"],
        "compiler_output": build_receipt["compiler_output"],
        "build_command": build_receipt["command"],
        "build_output_sha256": "sha256:"
        + hashlib.sha256(str(build_receipt.get("output", "")).encode()).hexdigest(),
        "PYO3_PYTHON": build_receipt["PYO3_PYTHON"],
        "CARGO_TARGET_DIR": build_receipt["CARGO_TARGET_DIR"],
        "actual_import_path": str(Path(binding.__file__).resolve()),
        "native_class": "carnot._rust.RustArchiveController7256",
        "native_methods": methods,
        "compiled_execution": True,
        "python_fallback_used": False,
        "shared_installed_extension_replaced": False,
    }


def _validation_receipt(name: str, command: Sequence[str], exit_code: int, log: str) -> JsonDict:
    """Bind one actual validation command and its complete captured output."""

    return {
        "name": name,
        "command": list(command),
        "exit_code": exit_code,
        "log_sha256": "sha256:" + hashlib.sha256(log.encode()).hexdigest(),
    }


def build_artifact(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Authenticate, build, replay, attack, reduce, and cold-check the prototype."""

    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    phase_started = time.monotonic()
    checks, sources = collect_preconditions(root, paths)
    spans["preconditions"] = time.monotonic() - phase_started
    hashes = dict(sources["hashes"])
    if gate_summary(checks)["passed"] is not True:
        artifact = _base_artifact(
            checks,
            hashes,
            paths,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - monotonic_start,
        )
        artifact["honest_verdict"] = "blocked_external:" + str(
            artifact["gate_check_summary"]["failed_check"]
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    progress(1, "start", "progress and heartbeat contract active")
    progress(1, "end", "all long calls have bounded external observation")
    progress(2, "start", "MODEL_SPECS empty; no model load or generation")
    progress(2, "end", "current model, load, generation, and inference counts are zero")
    progress(3, "start", "build and import isolated persistent native extension")
    phase_started = time.monotonic()
    extension, build_receipt = build_native_extension(root)
    progress(3, "before", "load exact compiled extension")
    binding = exp7230.load_native_extension(extension)
    progress(3, "after", f"loaded module_file={Path(binding.__file__).resolve()}")
    identity = _native_identity(extension, binding, build_receipt)
    hashes[str(extension.resolve())] = sha256_file(extension)
    hashes[str(Path(identity["wheel_path"]).resolve())] = identity["wheel_sha256"]
    spans["native_build_and_import"] = time.monotonic() - phase_started
    progress(3, "end", "compiler, wheel, module, and import receipts complete")

    progress(4, "before", "eight-stream three-controller differential replay")
    phase_started = time.monotonic()
    parity, checkpoints = run_differential_replay(binding)
    spans["differential_replay"] = time.monotonic() - phase_started
    progress(4, "after", f"completed_streams=8/8 parity_rows={len(parity)}")

    progress(5, "start", "E2E-003/004/007 adapted restore and transaction checks")
    phase_started = time.monotonic()
    fresh = run_fresh_process_continuation(extension, checkpoints[-1])
    negative_path = paths.negative_sidecar.with_name("experiment_7256_negative_durable.json")
    negatives = run_negative_controls(binding, negative_path)
    negative_payload = {
        "schema": "carnot.exp7256.negative_fixtures.v1",
        "injected_fixture": True,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "rows": negatives,
    }
    negative_receipt = atomic_write(paths.negative_sidecar, negative_payload)
    hashes[str(paths.negative_sidecar.resolve())] = negative_receipt["sha256"]
    spans["e2e_and_negative_controls"] = time.monotonic() - phase_started
    progress(5, "end", f"fresh_passed={fresh['passed']} negative_controls={len(negatives)}")

    progress(6, "before", "independent raw-row reduction; throughput benchmark deferred")
    phase_started = time.monotonic()
    reducer = reduce_parity_rows(parity)
    conversions = _conversion_rows(parity)
    spans["independent_reduction"] = time.monotonic() - phase_started
    progress(
        6,
        "after",
        f"event_arm_rows={reducer['event_arm_rows']} mismatches={reducer['event_mismatch_count']}",
    )

    persistent = next(row for row in conversions if row["arm"] == "persistent_native_controller")
    ready = int(
        reducer["event_mismatch_count"] == 0
        and reducer["declared_mismatch_count"] == 0
        and reducer["all_rollbacks_exact"] is True
        and fresh["passed"] is True
        and all(row["passed"] is True for row in negatives)
        and persistent["hot_path_json_parse_count"] == 0
        and persistent["active_reconstruction_count"] == 0
        and identity["compiled_execution"] is True
    )
    if ready != 1:
        raise RuntimeError("persistent native semantic parity or transaction gate failed")

    checkpoint_payload = {
        "schema": "carnot.exp7256.checkpoint.v1",
        "native_binary_receipt": identity,
        "parity_rows": parity,
        "conversion_count_rows": conversions,
        "fresh_process_receipt": fresh,
        "independent_reducer_receipt": reducer,
        "continuation_checkpoint": checkpoints[-1],
    }
    checkpoint_receipt = atomic_write(paths.checkpoint, checkpoint_payload)
    hashes[str(paths.checkpoint.resolve())] = checkpoint_receipt["sha256"]
    validation_receipts = [
        _validation_receipt(
            "isolated_native_build",
            build_receipt["command"],
            build_receipt["exit_code"],
            str(build_receipt.get("output", "")),
        ),
        _validation_receipt(
            "fresh_process_e2e",
            fresh["command"],
            fresh["exit_code"],
            str(fresh.get("stdout_sha256", "")) + str(fresh.get("stderr_sha256", "")),
        ),
        _validation_receipt(
            "independent_raw_row_reducer",
            ["in_process", "reduce_parity_rows"],
            0,
            canonical_json(reducer),
        ),
    ]
    baseline_validation_failures: list[dict[str, Any]] = []
    if paths.validation_sidecar.is_file():
        validation_sidecar = exp7243.read_object(paths.validation_sidecar)
        extra = validation_sidecar.get("validation_receipts", [])
        if isinstance(extra, list):
            validation_receipts.extend(row for row in extra if isinstance(row, dict))
        baseline = validation_sidecar.get("baseline_validation_failures", [])
        if isinstance(baseline, list):
            baseline_validation_failures.extend(row for row in baseline if isinstance(row, dict))
        hashes[str(paths.validation_sidecar.resolve())] = sha256_file(paths.validation_sidecar)

    artifact = _base_artifact(
        checks,
        hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": [*parity, *conversions, fresh, *negatives],
            "sample_size_budget": _sample_budget(True),
            "acceptance_gate_results": {
                "exact_three_arm_parity": {
                    "principle": "Any decision, state, archive, commit, or lineage mismatch blocks readiness.",
                    "expected": 0,
                    "observed": reducer["declared_mismatch_count"],
                    "passed": reducer["declared_mismatch_count"] == 0,
                },
                "persistent_native_ownership": {
                    "principle": "Typed hot-path events must not restore JSON or reconstruct an active object.",
                    "expected": {"hot_path_json_parse_count": 0, "active_reconstruction_count": 0},
                    "observed": {
                        "hot_path_json_parse_count": persistent["hot_path_json_parse_count"],
                        "active_reconstruction_count": persistent["active_reconstruction_count"],
                    },
                    "passed": persistent["hot_path_json_parse_count"] == 0
                    and persistent["active_reconstruction_count"] == 0,
                },
                "restart_and_transactions": {
                    "principle": "Fresh restore and rejected transactions must preserve exact decisions and bytes.",
                    "expected": True,
                    "observed": fresh["passed"] is True
                    and all(row["passed"] is True for row in negatives),
                    "passed": fresh["passed"] is True
                    and all(row["passed"] is True for row in negatives),
                },
            },
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive: exact oracle parity and persistent native ownership passed; throughput deferred to Exp7257",
            "validation_receipts": validation_receipts,
            "baseline_validation_failures": baseline_validation_failures,
            "native_controller_ready_score": ready,
            "native_binary_receipt": identity,
            "parity_rows": parity,
            "conversion_count_rows": conversions,
            "throughput_value_score": None,
            "fresh_process_receipt": fresh,
            "negative_control_rows": negatives,
            "independent_reducer_receipt": reducer,
            "checkpoint_receipt": checkpoint_receipt,
            "negative_fixture_receipt": negative_receipt,
            "upstream_receipt": {
                "path": str(UPSTREAM_RELATIVE),
                "sha256": hashes[str(UPSTREAM_RELATIVE)],
                "status": sources["upstream"].get("status"),
                "native_archive_ready_score": sources["upstream"].get("native_archive_ready_score"),
                "native_archive_cost_value_score": sources["upstream"].get(
                    "native_archive_cost_value_score"
                ),
                "historical_models_currently_invoked": False,
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, cold-validate, and atomically write the terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / NEGATIVE_RELATIVE,
        root / VALIDATION_RELATIVE,
        output,
    )
    artifact = build_artifact(root, paths)
    progress(7, "before", "cold terminal artifact validation")
    errors = validate_artifact(
        artifact, check_files=artifact.get("status") == "complete", root=root
    )
    progress(7, "after", f"cold terminal artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7256 artifact:{errors}")
    progress(8, "before", "atomic terminal artifact write")
    receipt = atomic_write(output, artifact)
    progress(8, "after", f"atomic terminal artifact write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and read-only validation option."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded study or validate existing bytes without mutation."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(7, "before", f"read-only validation path={args.validate}")
        try:
            value = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            progress(7, "after", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(7, "after", f"read-only validation errors={len(errors)}")
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


if __name__ == "__main__":
    raise SystemExit(main())
