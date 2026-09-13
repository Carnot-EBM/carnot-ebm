"""Measure complete archive-event cost across three shipped controller arms.

The benchmark uses one validation, serialization, durable-write, and restore
policy for every arm. This makes native ownership the only intended cost
difference. Exact parity is an oracle check and stays separate from cost value.

Spec refs: REQ-RUSTPY-7257 and SCENARIO-RUSTPY-7257-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import ModuleType
from typing import Any

from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7243_v637_native_memory as exp7243
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7257
SCHEMA = "carnot.exp7257.v638_native_cost.v1"
MILESTONE = "2026.09.638"
RUN_DATE = "20260913"
RANDOM_SEED = 7_257_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

ARCHIVE_CAPACITIES = (1, 4)
BATCH_SIZES = (1, 16, 128)
PAIRED_BLOCKS = 30
ARMS = ("python_reference", "old_native_wrapper", "persistent_native_controller")
COMPARISON_ARMS = ("old_native_wrapper", "persistent_native_controller")
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DURABILITY_POLICY = {
    "snapshot_encoding": "canonical_json_utf8",
    "file_fsync_per_block": 1,
    "atomic_rename_per_block": 1,
    "directory_fsync_per_block": 1,
    "restore_per_block": 1,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7257_v638_native_cost.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7257_v638_native_cost.json")
VALIDATION_RELATIVE = Path(
    "results/checkpoints/experiment_7257_v638_native_cost_validation_receipts.json"
)
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7257_v638_native_cost_storage")
UPSTREAM_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
UPSTREAM_PATH = REPO_ROOT / UPSTREAM_RELATIVE
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/rust-python-boundary/spec.md")
TARGET_RELATIVE = Path("target/experiment-7257-interpreter-bound")
LOAD_RELATIVE = Path("target/experiment-7257-load")
EXPECTED_EXP7256_SHA256 = "sha256:0afb88fecb2875fe4793e25dabc107fbc880b11da84bd965871b083e9c1ba4fc"

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("crates/carnot-python/src/experiment_7256_archive_controller.rs"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7243_v637_native_memory.py"),
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("python/carnot/experiment_7257_v638_native_cost.py"),
    Path("scripts/experiments/experiment_7257_v638_native_cost.py"),
    Path("tests/python/test_experiment_7257_v638_native_cost.py"),
    SPEC_RELATIVE,
    UPSTREAM_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260913 and retain actual UTC start and end timestamps.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name kv260, gatemate or polarfire.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked_* verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; a failed scientific acceptance gate forbids positive.",
    "validation_receipts": "Record actual command, exit code and log hash; no skipped, weakened, deleted or reverted tests.",
    "native_cost_complete_score": "One means all 540 block-arm cells and validation operations are accounted for.",
    "native_event_cost_value_score": "One requires both interactive-capacity cost gates and exact parity with equal durability.",
    "cost_rows": "Every capacity, batch, block, arm and full/component duration is retained.",
    "amortization_rows": "Include build, import, state transfer and break-even call counts.",
    "native_binary_receipt": "Verify the exact isolated binary and interpreter also used for parity.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7243.canonical_json
sha256_file = exp7243.sha256_file
artifact_checksum = exp7243.artifact_checksum
check = exp7243.check
gate_summary = exp7243.gate_summary
atomic_write = exp7243.atomic_write
_finish_row = exp7243._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional measurements separate from the terminal result."""

    checkpoint: Path
    validation_sidecar: Path
    storage_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the paths used by the public experiment command."""

        return cls(
            REPO_ROOT / CHECKPOINT_RELATIVE,
            REPO_ROOT / VALIDATION_RELATIVE,
            REPO_ROOT / STORAGE_RELATIVE,
            REPO_ROOT / RESULT_RELATIVE,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below one caller-owned temporary directory."""

        return cls(
            root / "checkpoints/exp7257.json",
            root / "checkpoints/validation.json",
            root / "checkpoints/storage",
            root / "experiment_7257.json",
        )

    def writable_targets(self) -> list[bool]:
        """Check output ownership without creating success-shaped bytes."""

        return [
            _writable(path)
            for path in (self.checkpoint, self.validation_sidecar, self.storage_dir, self.artifact)
        ]


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Flush a phase boundary and its real monotonic elapsed time."""

    suffix = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    print(f"[phase {phase} {boundary}] {operation}{suffix}", flush=True)


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output bytes."""

    parent = path if path.is_dir() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


_UPSTREAM_HELPER = r"""
import json
from pathlib import Path
import sys
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7243_v637_native_memory as exp7243

path = Path(sys.argv[1])
exclusion_path = Path(sys.argv[2])
artifact = json.loads(path.read_text(encoding="utf-8"))
try:
    checksum_valid = exp7243.artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
except (TypeError, ValueError):
    checksum_valid = False
identity = artifact.get("native_binary_receipt", {})
quarantine = exp7213.quarantine_state(
    artifact, exclusion_path.read_text(encoding="utf-8"), path.name, "exp7256-native-controller"
)
summary = {
    "upstream": {
        "status": artifact.get("status"),
        "native_controller_ready_score": artifact.get("native_controller_ready_score"),
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        "MODEL_SPECS": artifact.get("MODEL_SPECS"),
        "model_invoked": artifact.get("model_invoked"),
    },
    "checksum_valid": checksum_valid,
    "identity": identity,
    "quarantine": quarantine,
}
print("__CARNOT_JSON__" + json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
"""


def _read_upstream_summary(path: Path, exclusion_path: Path) -> JsonDict:
    """Authenticate the large source in a bounded process that releases its RSS."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _UPSTREAM_HELPER,
        str(path),
        str(exclusion_path),
    ]
    with exp7217._Heartbeat("Exp7257 upstream artifact authentication", interval_s=30):
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(REPO_ROOT / "python"),
            },
        )
    if completed.returncode != 0:
        return {
            "upstream": {},
            "checksum_valid": False,
            "identity": {},
            "quarantine": {
                "quarantined": True,
                "authentication_error": completed.stderr[-2_000:],
            },
        }
    for line in completed.stdout.splitlines():
        if line.startswith("__CARNOT_JSON__"):
            value = json.loads(line.removeprefix("__CARNOT_JSON__"))
            if isinstance(value, Mapping):
                return dict(value)
    return {
        "upstream": {},
        "checksum_valid": False,
        "identity": {},
        "quarantine": {"quarantined": True, "authentication_error": "missing_summary"},
    }


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate upstream bytes, quarantine, imports, tools, and outputs."""

    started = time.monotonic()
    progress(
        0, "start", "authenticate upstream bytes, quarantine, imports, and output paths", started
    )
    selected_upstream = upstream_path or root / UPSTREAM_RELATIVE
    summary = _read_upstream_summary(selected_upstream, root / EXCLUSION_RELATIVE)
    upstream = summary["upstream"]
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    selected_hash = sha256_file(selected_upstream) if selected_upstream.is_file() else None
    hashes[str(selected_upstream.resolve())] = selected_hash
    quarantine = summary["quarantine"]
    identity = summary["identity"]
    module_path = Path(str(identity.get("module_file", "")))
    observed_binary_hash = sha256_file(module_path) if module_path.is_file() else None
    declared_binary_hash = identity.get("module_sha256")
    checksum_valid = summary["checksum_valid"] is True
    spec_text = (root / SPEC_RELATIVE).read_text(encoding="utf-8")
    tools = {name: shutil.which(name) for name in ("cargo", "rustc", "ldd")}
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
            "REQ-RUSTPY-7257",
            True,
            "REQ-RUSTPY-7257" in spec_text,
            "REQ-RUSTPY-7257" in spec_text,
        ),
        check(
            "exp7256_artifact_hash",
            str(selected_upstream),
            "sha256",
            EXPECTED_EXP7256_SHA256,
            selected_hash,
            selected_hash == EXPECTED_EXP7256_SHA256,
        ),
        check(
            "exp7256_ready_complete_checksum",
            str(selected_upstream),
            "status,native_controller_ready_score,reproducibility_checksum",
            {"status": "complete", "ready": 1, "checksum": True},
            {
                "status": upstream.get("status"),
                "ready": upstream.get("native_controller_ready_score"),
                "checksum": checksum_valid,
            },
            upstream.get("status") == "complete"
            and upstream.get("native_controller_ready_score") == 1
            and checksum_valid,
        ),
        check(
            "exp7256_not_quarantined",
            str(selected_upstream),
            "quarantined",
            False,
            quarantine,
            quarantine.get("quarantined") is False,
        ),
        check(
            "exp7256_imported_binary",
            str(module_path),
            "module_sha256",
            declared_binary_hash,
            observed_binary_hash,
            bool(declared_binary_hash) and observed_binary_hash == declared_binary_hash,
        ),
        check(
            "imports_tools_and_outputs",
            "host",
            "imports,tools,owned paths",
            "all available and writable",
            {"tools": tools, "outputs": paths.writable_targets()},
            all(tools.values())
            and all(paths.writable_targets())
            and hasattr(exp7256, "PersistentNativeArchiveController"),
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
        "upstream": upstream,
        "quarantine": quarantine,
        "upstream_binary_hash": observed_binary_hash,
        "declared_binary_hash": declared_binary_hash,
        "upstream_binary_path": str(module_path),
    }


def seed_cost_state(capacity: int) -> JsonDict:
    """Reuse the shipped nonempty archive and validation-window fixture."""

    return exp7243._seed_cost_state(capacity)


def _cost_release(block: int, index: int) -> JsonDict:
    """Create one released event with a fixed nonzero feedback delay."""

    release = exp7243._cost_release(block, index)
    release["request_index"] = int(release["request_index"]) - 2
    return release


def _controller_from_state(binding: ModuleType, arm: str, state: Mapping[str, Any]) -> Any:
    """Restore one shipped controller without changing its ownership boundary."""

    if arm == "python_reference":
        return exp7256._CountingPythonArchive.from_state(state)
    if arm == "old_native_wrapper":
        return exp7256._CountingOldNativeArchive.from_state_with_binding(binding, state)
    if arm == "persistent_native_controller":
        return exp7256.PersistentNativeArchiveController.from_state_with_binding(binding, state)
    raise ValueError(f"unknown cost arm:{arm}")


def _measure_cost_arm(
    binding: ModuleType,
    arm: str,
    state: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    durable_path: Path,
) -> JsonDict:
    """Measure one full block with the same safety and durability operations."""

    controller = _controller_from_state(binding, arm, state)
    initial_bytes = transactional.canonical_json_bytes(dict(state))
    transactional._atomic_write(durable_path, initial_bytes)
    transcript: list[JsonDict] = []
    operation_hashes: list[str] = []
    components = {
        "lookup_ns": 0,
        "query_ns": 0,
        "delayed_release_update_ns": 0,
        "validation_ns": 0,
        "serialization_ns": 0,
        "durable_commit_ns": 0,
        "restore_ns": 0,
    }
    total_started = time.perf_counter_ns()
    for release in trace:
        event = {
            "event_id": release["event_id"],
            "family_id": release["family_id"],
            "numeric_value": release["numeric_value"],
        }
        started = time.perf_counter_ns()
        prediction = controller.predict(event)
        energies = [controller.energy(label, event) for label in ("accept", "reject")]
        components["lookup_ns"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        selected = controller.select_request([event], {str(event["event_id"]): 0})
        components["query_ns"] += time.perf_counter_ns() - started
        parent_hash = controller.state_hash()
        started = time.perf_counter_ns()
        receipt = controller.commit_batch(
            [release],
            current_cycle=int(release["release_index"]),
            expected_parent_hash=parent_hash,
        )
        components["delayed_release_update_ns"] += time.perf_counter_ns() - started
        transcript.append(
            {
                "event_id": event["event_id"],
                "prediction": prediction,
                "energies": energies,
                "selected_event_id": selected["event_id"],
                "parent_hash": parent_hash,
                "child_hash": receipt["new_state_hash"],
            }
        )
        operation_hashes.append(
            "sha256:" + hashlib.sha256(canonical_json(receipt["operations"]).encode()).hexdigest()
        )

    started = time.perf_counter_ns()
    serialized = controller.state_bytes()
    components["serialization_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    decoded = json.loads(serialized)
    validated = exp7240.ArchivedBeliefController.from_state(decoded)
    validated_bytes = validated.state_bytes()
    components["validation_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    durable_receipt = transactional._atomic_write(durable_path, serialized)
    components["durable_commit_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    durable_bytes = durable_path.read_bytes()
    restored = _controller_from_state(binding, arm, json.loads(durable_bytes))
    last_event = {
        "event_id": trace[-1]["event_id"],
        "family_id": trace[-1]["family_id"],
        "numeric_value": trace[-1]["numeric_value"],
    }
    restored_probe = restored.predict(last_event)
    restored_hash = restored.state_hash()
    components["restore_ns"] = time.perf_counter_ns() - started
    total = time.perf_counter_ns() - total_started
    accounted = sum(components.values())
    components["residual_unaccounted_ns"] = max(total - accounted, 0)
    state_hash = controller.state_hash()
    safety = (
        serialized == validated_bytes
        and serialized == durable_bytes
        and state_hash == restored_hash
        and restored_probe == controller.predict(last_event)
        and durable_receipt
        == {
            "file_fsync": True,
            "rename": True,
            "directory_fsync": True,
        }
    )
    native_timing = controller.native_timing() if arm == "old_native_wrapper" else {}
    conversions = controller.conversion_counts() if arm == "persistent_native_controller" else {}
    return {
        "total_block_ns": total,
        "total_event_ns": total / len(trace),
        "component_ns": components,
        "decision_count": len(transcript),
        "update_count": len(trace),
        "decision_receipt_sha256": "sha256:"
        + hashlib.sha256(canonical_json(transcript).encode()).hexdigest(),
        "operation_receipt_sha256": "sha256:"
        + hashlib.sha256(canonical_json(operation_hashes).encode()).hexdigest(),
        "serialized_sha256": "sha256:" + hashlib.sha256(serialized).hexdigest(),
        "durable_sha256": "sha256:" + hashlib.sha256(durable_bytes).hexdigest(),
        "final_state_hash": state_hash,
        "restored_state_hash": restored_hash,
        "durability_receipt": durable_receipt,
        "durable_commit_count": 1,
        "safety_checks_passed": safety,
        "native_kernel_ns": int(native_timing.get("native_kernel_ns", 0)),
        "native_call_count": int(native_timing.get("native_call_count", 0)),
        "conversion_counts": dict(conversions),
        "serialized_bytes": len(serialized),
    }


def run_cost_benchmark(
    binding: ModuleType,
    *,
    storage_dir: Path,
    capacities: Sequence[int] = ARCHIVE_CAPACITIES,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
    seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Run fixed warmup, then interleave all complete three-arm blocks."""

    if not capacities or not batch_sizes or blocks <= 0:
        raise ValueError("cost roster must be nonempty")
    storage_dir.mkdir(parents=True, exist_ok=True)
    durable_path = storage_dir / "durable_state.json"
    states = {capacity: seed_cost_state(capacity) for capacity in capacities}
    traces = {
        (batch, block): [_cost_release(block, index) for index in range(batch)]
        for batch in batch_sizes
        for block in range(blocks)
    }
    benchmark_started = time.monotonic()
    progress(
        6, "before", "separate full-event warmup; no warmup row is retained", benchmark_started
    )
    for capacity in capacities:
        for batch in batch_sizes:
            for arm in ARMS:
                _measure_cost_arm(binding, arm, states[capacity], traces[(batch, 0)], durable_path)
    progress(6, "after", "separate full-event warmup complete", benchmark_started)

    rng = random.Random(seed)
    orders: dict[tuple[int, int, int], list[str]] = {}
    for capacity in capacities:
        for batch in batch_sizes:
            for block in range(blocks):
                order = list(ARMS)
                rng.shuffle(order)
                orders[(capacity, batch, block)] = order

    rows: list[JsonDict] = []
    completed_blocks = 0
    planned_blocks = len(capacities) * len(batch_sizes) * blocks
    last_report = time.monotonic()
    progress(6, "before", f"interleaved paired blocks planned={planned_blocks}", benchmark_started)
    for capacity in capacities:
        for batch in batch_sizes:
            for block in range(blocks):
                trace = traces[(batch, block)]
                trace_hash = "sha256:" + hashlib.sha256(canonical_json(trace).encode()).hexdigest()
                measured = {
                    arm: _measure_cost_arm(binding, arm, states[capacity], trace, durable_path)
                    for arm in orders[(capacity, batch, block)]
                }
                reference = measured["python_reference"]
                for order_index, arm in enumerate(orders[(capacity, batch, block)]):
                    result = measured[arm]
                    mismatch_count = sum(
                        (
                            result[name] != reference[name]
                            for name in (
                                "decision_receipt_sha256",
                                "operation_receipt_sha256",
                                "serialized_sha256",
                                "final_state_hash",
                                "restored_state_hash",
                            )
                        )
                    )
                    rows.append(
                        _finish_row(
                            {
                                "unit_id": f"cost:{capacity}:{batch}:{block}:{arm}",
                                "arm": arm,
                                "seed": seed,
                                "metric": result["total_event_ns"],
                                "error": None,
                                "abstention": False,
                                "censored": False,
                                "archive_capacity": capacity,
                                "batch_size": batch,
                                "block": block,
                                "arm_order": order_index,
                                "trace_sha256": trace_hash,
                                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                                "thread_limits": dict(THREAD_LIMITS),
                                "storage_device": durable_path.stat().st_dev,
                                "durability_policy": deepcopy(DURABILITY_POLICY),
                                "warmup_included": False,
                                "parity_mismatch_count": mismatch_count,
                                **result,
                            }
                        )
                    )
                completed_blocks += 1
                now = time.monotonic()
                if now - last_report >= 60:
                    print(
                        f"[phase 6 progress] completed_blocks={completed_blocks}/{planned_blocks} "
                        f"completed_arm_rows={len(rows)}/{planned_blocks * len(ARMS)} "
                        f"elapsed_s={now - benchmark_started:.3f}",
                        flush=True,
                    )
                    last_report = now
    progress(
        6,
        "after",
        f"completed_arm_rows={len(rows)}/{planned_blocks * len(ARMS)}",
        benchmark_started,
    )
    return rows


def _bootstrap_interval(values: Sequence[float], seed: int, draws: int = 10_000) -> list[float]:
    """Return the fixed paired bootstrap 95 percent mean interval."""

    if not values:
        raise ValueError("paired ratios are empty")
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def reduce_cost_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacities: Sequence[int] = ARCHIVE_CAPACITIES,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
) -> JsonDict:
    """Independently recompute every paired interval and frozen cost gate."""

    expected_rows = len(capacities) * len(batch_sizes) * blocks * len(ARMS)
    if len(rows) != expected_rows:
        raise ValueError(f"incomplete cost roster:{len(rows)}/{expected_rows}")
    cells = []
    for capacity in capacities:
        for batch in batch_sizes:
            for comparison in COMPARISON_ARMS:
                ratios = []
                for block in range(blocks):
                    pair = {
                        str(row["arm"]): float(row["total_event_ns"])
                        for row in rows
                        if row.get("archive_capacity") == capacity
                        and row.get("batch_size") == batch
                        and row.get("block") == block
                    }
                    if set(pair) != set(ARMS):
                        raise ValueError(f"incomplete cost roster:{capacity}:{batch}:{block}")
                    ratios.append(pair["python_reference"] / pair[comparison])
                interval = _bootstrap_interval(
                    ratios,
                    RANDOM_SEED
                    + capacity * 10_000
                    + batch * 10
                    + COMPARISON_ARMS.index(comparison),
                )
                cells.append(
                    {
                        "archive_capacity": capacity,
                        "batch_size": batch,
                        "comparison": f"python_reference_over_{comparison}",
                        "paired_blocks": blocks,
                        "mean_speedup": sum(ratios) / len(ratios),
                        "ci95": interval,
                        "paired_ratios": ratios,
                    }
                )
    batch_one_gates = {
        str(capacity): next(
            cell["ci95"][0] > 1.0
            for cell in cells
            if cell["archive_capacity"] == capacity
            and cell["batch_size"] == 1
            and cell["comparison"] == "python_reference_over_persistent_native_controller"
        )
        for capacity in capacities
    }
    parity_failures = sum(int(row.get("parity_mismatch_count", 0)) for row in rows)
    safety_failures = sum(int(row.get("safety_checks_passed") is not True) for row in rows)
    durability_equal = all(
        row.get("durability_policy") == DURABILITY_POLICY
        and row.get("durability_receipt")
        == {"file_fsync": True, "rename": True, "directory_fsync": True}
        and row.get("durable_commit_count") == 1
        for row in rows
    )
    interactive_lowers = [
        cell["ci95"][0]
        for cell in cells
        if cell["batch_size"] == 1
        and cell["comparison"] == "python_reference_over_persistent_native_controller"
    ]
    complete = int(
        len(rows) == expected_rows
        and parity_failures == 0
        and safety_failures == 0
        and durability_equal
    )
    value = int(complete == 1 and all(batch_one_gates.values()))
    return {
        "cost_row_count": len(rows),
        "expected_cost_row_count": expected_rows,
        "cells": cells,
        "batch_one_capacity_gates": batch_one_gates,
        "interactive_persistent_lower_ci95_min": min(interactive_lowers),
        "parity_failure_count": parity_failures,
        "safety_failure_count": safety_failures,
        "equal_durability": durability_equal,
        "native_cost_complete_score": complete,
        "native_event_cost_value_score": value,
        "nfr_01_10x_met": complete == 1 and min(interactive_lowers) >= 10.0,
        "research_program_100x_met": complete == 1 and min(interactive_lowers) >= 100.0,
    }


def synthetic_cost_rows(
    *,
    python_ns: int,
    old_native_ns: int,
    persistent_ns: int,
) -> list[JsonDict]:
    """Create the full deterministic row shape for reducer and validator tests."""

    durations = {
        "python_reference": python_ns,
        "old_native_wrapper": old_native_ns,
        "persistent_native_controller": persistent_ns,
    }
    rows = []
    for capacity in ARCHIVE_CAPACITIES:
        for batch in BATCH_SIZES:
            for block in range(PAIRED_BLOCKS):
                for order, arm in enumerate(ARMS):
                    duration = durations[arm]
                    rows.append(
                        _finish_row(
                            {
                                "unit_id": f"fixture:{capacity}:{batch}:{block}:{arm}",
                                "arm": arm,
                                "seed": RANDOM_SEED,
                                "metric": duration,
                                "error": None,
                                "abstention": False,
                                "censored": False,
                                "archive_capacity": capacity,
                                "batch_size": batch,
                                "block": block,
                                "arm_order": order,
                                "trace_sha256": "sha256:fixture",
                                "cpu_affinity": [0],
                                "thread_limits": dict(THREAD_LIMITS),
                                "storage_device": 1,
                                "durability_policy": deepcopy(DURABILITY_POLICY),
                                "warmup_included": False,
                                "parity_mismatch_count": 0,
                                "total_block_ns": duration * batch,
                                "total_event_ns": duration,
                                "component_ns": {
                                    "lookup_ns": 1,
                                    "query_ns": 1,
                                    "delayed_release_update_ns": 1,
                                    "validation_ns": 1,
                                    "serialization_ns": 1,
                                    "durable_commit_ns": 1,
                                    "restore_ns": 1,
                                    "residual_unaccounted_ns": max(duration * batch - 7, 0),
                                },
                                "decision_count": batch,
                                "update_count": batch,
                                "decision_receipt_sha256": "sha256:fixture",
                                "operation_receipt_sha256": "sha256:fixture",
                                "serialized_sha256": "sha256:fixture",
                                "durable_sha256": "sha256:fixture",
                                "final_state_hash": "sha256:fixture",
                                "restored_state_hash": "sha256:fixture",
                                "durability_receipt": {
                                    "file_fsync": True,
                                    "rename": True,
                                    "directory_fsync": True,
                                },
                                "durable_commit_count": 1,
                                "safety_checks_passed": True,
                                "native_kernel_ns": 0,
                                "native_call_count": 0,
                                "conversion_counts": {},
                                "serialized_bytes": 1,
                            }
                        )
                    )
    return rows


_FRESH_HELPER = r"""
import importlib.util
import json
from pathlib import Path
import sys
from carnot import experiment_7257_v638_native_cost as experiment

extension = Path(sys.argv[1]).resolve()
snapshot = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("carnot._rust", extension)
if spec is None or spec.loader is None:
    raise RuntimeError("native loader unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = json.loads(snapshot.read_text(encoding="utf-8"))
controller = experiment.exp7256.PersistentNativeArchiveController.from_state_with_binding(module, state)
probe = {"event_id": "exp7257-fresh-probe", "family_id": "lower_bound", "numeric_value": 9}
result = {
    "decision": controller.predict(probe),
    "energies": [controller.energy(label, probe) for label in ("accept", "reject")],
    "state_hash": controller.state_hash(),
    "module_file": str(Path(module.__file__).resolve()),
    "module_sha256": experiment.sha256_file(extension),
    "interpreter": str(Path(sys.executable).absolute()),
}
print("__CARNOT_JSON__" + json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
"""


def run_fresh_process_replay(extension: Path, snapshot: Path) -> JsonDict:
    """Run E2E-003/004 against the actual timed binary and snapshot path."""

    started = time.monotonic()
    progress(5, "before", "fresh-process import and snapshot replay", started)
    state = json.loads(snapshot.read_text(encoding="utf-8"))
    reference = exp7240.ArchivedBeliefController.from_state(state)
    probe = {"event_id": "exp7257-fresh-probe", "family_id": "lower_bound", "numeric_value": 9}
    expected = {
        "decision": reference.predict(probe),
        "energies": [reference.energy(label, probe) for label in ("accept", "reject")],
        "state_hash": reference.state_hash(),
    }
    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _FRESH_HELPER,
        str(extension),
        str(snapshot),
    ]
    with exp7217._Heartbeat("Exp7257 fresh-process snapshot replay", interval_s=30):
        completed = subprocess.run(
            command,
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
    same_decisions = (
        observed.get("decision") == list(expected["decision"])
        and observed.get("energies") == expected["energies"]
    )
    same_hash = observed.get("state_hash") == expected["state_hash"]
    passed = (
        completed.returncode == 0
        and same_decisions
        and same_hash
        and observed.get("module_file") == str(extension.resolve())
        and observed.get("module_sha256") == sha256_file(extension)
    )
    receipt = {
        "command": command,
        "exit_code": completed.returncode,
        "duration_s": time.monotonic() - started,
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_sha256": sha256_file(snapshot),
        "module_file": observed.get("module_file"),
        "module_sha256": observed.get("module_sha256"),
        "same_decisions": same_decisions,
        "same_state_hash": same_hash,
        "passed": passed,
    }
    progress(5, "after", f"fresh_process_passed={passed}", started)
    return receipt


def amortization_rows(
    costs: Sequence[Mapping[str, Any]], setup: Mapping[str, float]
) -> list[JsonDict]:
    """Keep cold setup and break-even estimates outside steady-state rows."""

    python_values = [
        float(row["total_event_ns"])
        for row in costs
        if row.get("batch_size") == 1 and row.get("arm") == "python_reference"
    ]
    persistent_values = [
        float(row["total_event_ns"])
        for row in costs
        if row.get("batch_size") == 1 and row.get("arm") == "persistent_native_controller"
    ]
    saving = sum(python_values) / len(python_values) - sum(persistent_values) / len(
        persistent_values
    )
    cold_s = sum(
        float(setup[name])
        for name in (
            "startup_duration_s",
            "build_duration_s",
            "import_duration_s",
            "state_transfer_duration_s",
        )
    )
    terms = [
        ("process_startup", setup["startup_duration_s"]),
        ("isolated_build", setup["build_duration_s"]),
        ("native_import", setup["import_duration_s"]),
        ("initial_state_transfer", setup["state_transfer_duration_s"]),
    ]
    rows = [
        {
            "term": term,
            "duration_s": float(duration),
            "included_in_steady_state": False,
            "break_even_calls": None,
        }
        for term, duration in terms
    ]
    rows.append(
        {
            "term": "persistent_native_break_even",
            "duration_s": cold_s,
            "included_in_steady_state": False,
            "mean_batch_one_saving_ns": saving,
            "break_even_calls": None if saving <= 0 else cold_s * 1_000_000_000 / saving,
        }
    )
    return rows


def _sample_budget(complete: bool) -> JsonDict:
    """Declare all 540 fixed block-arm rows and the fixed stopping rule."""

    planned = len(ARCHIVE_CAPACITIES) * len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS)
    return {
        "planned_independent_block_arm_units": planned,
        "attempted_independent_block_arm_units": planned if complete else 0,
        "completed_independent_block_arm_units": planned if complete else 0,
        "censored_independent_block_arm_units": 0 if complete else planned,
        "paired_blocks_per_cell": PAIRED_BLOCKS,
        "stopping_rule": "exactly 30 predeclared blocks per capacity and batch cell; no outcome extension",
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
        "random_seed": RANDOM_SEED,
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
        "native_cost_complete_score": 0,
        "native_event_cost_value_score": 0,
        "cost_rows": [],
        "cost_summary": {},
        "amortization_rows": [],
        "native_binary_receipt": {},
        "parity_rows": [],
        "fresh_process_receipt": {},
        "independent_raw_row_reducer": {},
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None},
        "methodology": {
            "task": "complete three-arm archive event cost",
            "inference": "no LLM invocation",
            "durability_policy": deepcopy(DURABILITY_POLICY),
        },
        "retirement_scope": "cost result not yet measured",
        "publication_performed": False,
        "default_pipeline_modified": False,
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return a complete row-free external-block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["honest_verdict"] = f"blocked_external:{failed.get('check')}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _parity_rows_from_costs(costs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce per-block checks to one explicit row for each controller arm."""

    rows = []
    for arm in ARMS:
        selected = [row for row in costs if row.get("arm") == arm]
        mismatches = sum(int(row.get("parity_mismatch_count", 0)) for row in selected)
        rows.append(
            _finish_row(
                {
                    "unit_id": f"parity:{arm}",
                    "arm": arm,
                    "seed": RANDOM_SEED,
                    "metric": mismatches,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "checked_cost_rows": len(selected),
                    "mismatch_count": mismatches,
                    "passed": mismatches == 0
                    and all(row.get("safety_checks_passed") is True for row in selected),
                }
            )
        )
    return rows


def _fixture_parity_rows() -> list[JsonDict]:
    """Return compact exact parity rows for closed validator tests."""

    return _parity_rows_from_costs(
        synthetic_cost_rows(python_ns=3, old_native_ns=2, persistent_ns=1)
    )


def _validation_receipt(name: str, command: Sequence[str], exit_code: int, log: str) -> JsonDict:
    """Bind one executed command to its exit code and output hash."""

    return {
        "name": name,
        "command": list(command),
        "exit_code": exit_code,
        "log_sha256": "sha256:" + hashlib.sha256(log.encode()).hexdigest(),
    }


def complete_artifact_fixture_for_test(*, cost_pass: bool) -> JsonDict:
    """Create one complete positive-cost or null-cost validator fixture."""

    costs = synthetic_cost_rows(
        python_ns=300 if cost_pass else 100,
        old_native_ns=200 if cost_pass else 80,
        persistent_ns=100 if cost_pass else 120,
    )
    summary = reduce_cost_rows(costs)
    parity = _parity_rows_from_costs(costs)
    failed = check("fixture", "fixture", "fixture", True, True, True)
    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    setup = {
        "startup_duration_s": 0.01,
        "build_duration_s": 0.1,
        "import_duration_s": 0.01,
        "state_transfer_duration_s": 0.01,
    }
    value = summary["native_event_cost_value_score"]
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": [*parity, *costs],
            "sample_size_budget": _sample_budget(True),
            "acceptance_gate_results": _acceptance_gates(summary),
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: exact parity and both interactive native cost gates passed"
                if value
                else "complete_null: exact parity passed but interactive native cost value failed"
            ),
            "validation_receipts": [_validation_receipt("fixture", ["fixture"], 0, "ok")],
            "native_cost_complete_score": summary["native_cost_complete_score"],
            "native_event_cost_value_score": value,
            "cost_rows": costs,
            "cost_summary": summary,
            "amortization_rows": amortization_rows(costs, setup),
            "native_binary_receipt": {
                "interpreter": str(Path(sys.executable).absolute()),
                "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
                "module_sha256": "sha256:fixture",
                "native_class": "carnot._rust.RustArchiveController7256",
                "compiled_execution": True,
                "python_fallback_used": False,
            },
            "parity_rows": parity,
            "fresh_process_receipt": {
                "passed": True,
                "same_decisions": True,
                "same_state_hash": True,
            },
            "independent_raw_row_reducer": summary,
            "retirement_scope": (
                "no retirement; measured boundary cost gate passed"
                if value
                else "retire this boundary optimization only; Rust and sampler families remain active"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _acceptance_gates(summary: Mapping[str, Any]) -> JsonDict:
    """Express each frozen criterion with expected, observed, and passed values."""

    interactive = summary["batch_one_capacity_gates"]
    return {
        "complete_540_row_roster": {
            "principle": "All planned blocks and safety operations must be accounted for.",
            "expected": 540,
            "observed": summary["cost_row_count"],
            "passed": summary["native_cost_complete_score"] == 1,
        },
        "interactive_capacity_one": {
            "principle": "Batch-one Python over persistent-native lower CI95 must exceed one.",
            "expected": ">1",
            "observed": interactive["1"],
            "passed": interactive["1"] is True,
        },
        "interactive_capacity_four": {
            "principle": "Batch-one Python over persistent-native lower CI95 must exceed one.",
            "expected": ">1",
            "observed": interactive["4"],
            "passed": interactive["4"] is True,
        },
        "exact_parity": {
            "principle": "A safety mismatch blocks native value.",
            "expected": 0,
            "observed": summary["parity_failure_count"],
            "passed": summary["parity_failure_count"] == 0,
        },
        "equal_durability": {
            "principle": "All arms must use one fsync, rename, directory fsync, and restore per block.",
            "expected": True,
            "observed": summary["equal_durability"],
            "passed": summary["equal_durability"] is True,
        },
        "nfr_01_10x": {
            "principle": "NFR-01 remains unmet unless both interactive lower bounds reach 10x.",
            "expected": ">=10",
            "observed": summary["interactive_persistent_lower_ci95_min"],
            "passed": summary["nfr_01_10x_met"] is True,
        },
        "self_learning_100x": {
            "principle": "The research aspiration remains unmet unless both interactive lower bounds reach 100x.",
            "expected": ">=100",
            "observed": summary["interactive_persistent_lower_ci95_min"],
            "passed": summary["research_program_100x_met"] is True,
        },
    }


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Recompute one row hash without trusting its stored digest."""

    material = dict(row)
    stored = material.pop("row_sha256", None)
    expected = "sha256:" + hashlib.sha256(canonical_json(material).encode()).hexdigest()
    return stored == expected


def validate_artifact(
    artifact: Mapping[str, Any], *, check_files: bool = False, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check identity, row shape, reducer, gates, receipts, and bytes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS.issubset(artifact), "missing_fields")
    if not REQUIRED_FIELDS.issubset(artifact):
        return errors
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        "identity",
    )
    add(artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE, "run_date")
    add(
        artifact.get("execution_venue") != EXECUTION_VENUE or not artifact.get("execution_host"),
        "execution_venue",
    )
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_invocation",
    )
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    add(not checksum_valid, "reproducibility_checksum")
    if artifact.get("status") == "blocked":
        gate = artifact.get("gate_check_summary", {})
        add(artifact.get("verdict_class") != "blocked", "blocked_class")
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run",
            "blocked_substrate",
        )
        add(
            any(artifact.get(name) for name in ("rows", "cost_rows", "parity_rows")), "blocked_rows"
        )
        add(not isinstance(gate, Mapping) or gate.get("passed") is not False, "blocked_gate")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        return errors

    costs = artifact.get("cost_rows", [])
    parity = artifact.get("parity_rows", [])
    rows = artifact.get("rows", [])
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    add(
        not isinstance(costs, list)
        or len(costs) != 540
        or any(
            row.get("safety_checks_passed") is not True
            or row.get("parity_mismatch_count") != 0
            or row.get("durability_policy") != DURABILITY_POLICY
            for row in costs
        ),
        "cost_rows",
    )
    add(
        not isinstance(parity, list)
        or len(parity) != 3
        or any(row.get("passed") is not True or row.get("mismatch_count") != 0 for row in parity),
        "parity_rows",
    )
    add(
        not isinstance(rows, list)
        or len(rows) != 543
        or any(not isinstance(row, Mapping) or not _row_hash_valid(row) for row in rows),
        "rows",
    )
    try:
        summary = reduce_cost_rows(costs)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        summary = None
    add(
        summary is None
        or artifact.get("cost_summary") != summary
        or artifact.get("independent_raw_row_reducer") != summary,
        "raw_row_reducer",
    )
    expected_complete = 0 if summary is None else summary["native_cost_complete_score"]
    expected_value = 0 if summary is None else summary["native_event_cost_value_score"]
    add(artifact.get("native_cost_complete_score") != expected_complete, "complete_score")
    add(artifact.get("native_event_cost_value_score") != expected_value, "cost_value_score")
    expected_class = "circular_positive" if expected_value == 1 else "null"
    add(
        artifact.get("verdict_class") != expected_class
        or artifact.get("verdict_class") == "positive",
        "verdict_class",
    )
    add(not str(artifact.get("honest_verdict", "")).startswith("complete_"), "honest_verdict")
    add(
        summary is None or artifact.get("acceptance_gate_results") != _acceptance_gates(summary),
        "acceptance_gate_results",
    )
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("completed_independent_block_arm_units") != 540
        or budget.get("censored_independent_block_arm_units") != 0,
        "sample_size_budget",
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
    identity = artifact.get("native_binary_receipt", {})
    add(
        not isinstance(identity, Mapping)
        or identity.get("compiled_execution") is not True
        or identity.get("python_fallback_used") is not False
        or identity.get("native_class") != "carnot._rust.RustArchiveController7256"
        or not str(identity.get("module_file", "")).endswith(
            str(sysconfig.get_config_var("EXT_SUFFIX"))
        )
        or not str(identity.get("module_sha256", "")).startswith("sha256:"),
        "native_binary_receipt",
    )
    fresh = artifact.get("fresh_process_receipt", {})
    add(
        not isinstance(fresh, Mapping)
        or fresh.get("passed") is not True
        or fresh.get("same_decisions") is not True
        or fresh.get("same_state_hash") is not True,
        "fresh_process_receipt",
    )
    if check_files:
        for path_text, expected_hash in artifact.get("source_artifact_hashes", {}).items():
            path = Path(path_text)
            resolved = path if path.is_absolute() else root / path
            add(
                not resolved.is_file() or sha256_file(resolved) != expected_hash,
                "source_artifact_hashes",
            )
        module = Path(str(identity.get("module_file", "")))
        add(
            not module.is_file() or sha256_file(module) != identity.get("module_sha256"),
            "native_module_hash",
        )
    return errors


def _build_timed_extension(root: Path) -> tuple[Path, JsonDict]:
    """Build only the existing PyO3 crate in an isolated Exp7257 target."""

    target = root / TARGET_RELATIVE
    environment = exp7217.interpreter_build_environment(Path(sys.executable), target)
    started = time.monotonic()
    progress(3, "before", "isolated carnot-python build subprocess", started)
    receipt = exp7217._stream_process(
        ["cargo", "build", "--release", "-p", "carnot-python"],
        root=root,
        environment=environment,
        operation="Exp7257 isolated carnot-python build",
        timeout_s=900,
    )
    build_duration = time.monotonic() - started
    library = target / "release/libcarnot_python.so"
    if not library.is_file():
        raise RuntimeError(f"native build output missing:{library}")
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
    receipt.update(
        {
            "build_duration_s": build_duration,
            "PYO3_PYTHON": environment["PYO3_PYTHON"],
            "CARGO_TARGET_DIR": environment["CARGO_TARGET_DIR"],
            "built_library": str(library.resolve()),
            "built_library_sha256": sha256_file(library),
            "loaded_copy": str(destination.resolve()),
            "module_sha256": sha256_file(destination),
        }
    )
    progress(3, "after", f"isolated build selected={destination.resolve()}", started)
    return destination, receipt


def _accept_validation_sidecar(path: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load only structured validation receipts prepared before measurement."""

    if not path.is_file():
        return [], []
    value = exp7243.read_object(path)
    receipts = value.get("validation_receipts", [])
    baseline = value.get("baseline_validation_failures", [])
    return (
        [dict(row) for row in receipts if isinstance(row, Mapping)]
        if isinstance(receipts, list)
        else [],
        [dict(row) for row in baseline if isinstance(row, Mapping)]
        if isinstance(baseline, list)
        else [],
    )


def build_artifact(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Authenticate, build, measure, replay, reduce, and classify the study."""

    invocation_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    phase_spans: JsonDict = {}
    phase_started = time.monotonic()
    checks, evidence = collect_preconditions(root, paths)
    phase_spans["preconditions"] = time.monotonic() - phase_started
    hashes = dict(evidence["hashes"])
    if gate_summary(checks)["passed"] is not True:
        artifact = _base_artifact(
            checks,
            hashes,
            paths,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - invocation_started,
        )
        artifact["honest_verdict"] = "blocked_external:" + str(
            artifact["gate_check_summary"]["failed_check"]
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    progress(1, "start", "progress and external heartbeat contract active", invocation_started)
    progress(1, "end", "all long operations expose real completion counts", invocation_started)
    progress(2, "start", "MODEL_SPECS empty; no model load or generation", invocation_started)
    progress(2, "end", "model_loads=0 generations=0 inference_calls=0", invocation_started)
    for name, value in THREAD_LIMITS.items():
        os.environ[name] = value

    setup: dict[str, float] = {}
    phase_started = time.monotonic()
    startup_command = [str(Path(sys.executable).absolute()), "-u", "-c", "pass"]
    progress(3, "before", "fresh interpreter startup subprocess", invocation_started)
    startup_started = time.monotonic()
    startup = subprocess.run(
        startup_command, capture_output=True, text=True, check=False, timeout=30
    )
    setup["startup_duration_s"] = time.monotonic() - startup_started
    progress(
        3, "after", f"fresh interpreter startup exit_code={startup.returncode}", invocation_started
    )
    extension, build_receipt = _build_timed_extension(root)
    setup["build_duration_s"] = float(build_receipt["build_duration_s"])
    progress(3, "before", "load exact isolated compiled extension", invocation_started)
    import_started = time.monotonic()
    binding = exp7230.load_native_extension(extension)
    setup["import_duration_s"] = time.monotonic() - import_started
    progress(
        3, "after", f"loaded module_file={Path(binding.__file__).resolve()}", invocation_started
    )
    transfer_started = time.monotonic()
    exp7256.PersistentNativeArchiveController.from_state_with_binding(binding, seed_cost_state(4))
    setup["state_transfer_duration_s"] = time.monotonic() - transfer_started
    hashes[str(extension.resolve())] = sha256_file(extension)
    phase_spans["startup_build_import_transfer"] = time.monotonic() - phase_started
    identity = {
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "module_file": str(Path(binding.__file__).resolve()),
        "module_sha256": sha256_file(extension),
        "upstream_module_file": evidence["upstream_binary_path"],
        "upstream_module_sha256": evidence["upstream_binary_hash"],
        "same_bytes_as_authenticated_upstream": sha256_file(extension)
        == evidence["upstream_binary_hash"],
        "build_command": build_receipt["command"],
        "build_exit_code": build_receipt["exit_code"],
        "build_output_sha256": "sha256:"
        + hashlib.sha256(str(build_receipt.get("output", "")).encode()).hexdigest(),
        "PYO3_PYTHON": build_receipt["PYO3_PYTHON"],
        "CARGO_TARGET_DIR": build_receipt["CARGO_TARGET_DIR"],
        "native_class": "carnot._rust.RustArchiveController7256",
        "compiled_execution": True,
        "python_fallback_used": False,
    }

    phase_started = time.monotonic()
    costs = run_cost_benchmark(binding, storage_dir=paths.storage_dir)
    phase_spans["paired_cost_benchmark"] = time.monotonic() - phase_started
    summary = reduce_cost_rows(costs)
    parity = _parity_rows_from_costs(costs)
    checkpoint_payload = {
        "schema": "carnot.exp7257.checkpoint.v1",
        "status": "provisional_measurement_complete",
        "selected_extension": str(extension.resolve()),
        "selected_extension_sha256": sha256_file(extension),
        "cost_rows": costs,
        "independent_raw_row_reducer": summary,
    }
    checkpoint_receipt = atomic_write(paths.checkpoint, checkpoint_payload)
    hashes[str(paths.checkpoint.resolve())] = checkpoint_receipt["sha256"]
    snapshot = paths.storage_dir / "durable_state.json"
    phase_started = time.monotonic()
    fresh = run_fresh_process_replay(extension, snapshot)
    phase_spans["fresh_process_e2e_003_004"] = time.monotonic() - phase_started

    complete = int(
        summary["native_cost_complete_score"] == 1
        and fresh["passed"] is True
        and identity["same_bytes_as_authenticated_upstream"] is True
    )
    value = int(complete == 1 and summary["native_event_cost_value_score"] == 1)
    external_receipts, baseline_failures = _accept_validation_sidecar(paths.validation_sidecar)
    validation_receipts = [
        _validation_receipt(
            "fresh_interpreter_startup",
            startup_command,
            startup.returncode,
            startup.stdout + startup.stderr,
        ),
        _validation_receipt(
            "isolated_native_build",
            build_receipt["command"],
            build_receipt["exit_code"],
            str(build_receipt.get("output", "")),
        ),
        _validation_receipt(
            "fresh_process_e2e_003_004",
            fresh["command"],
            fresh["exit_code"],
            fresh["stdout_sha256"] + fresh["stderr_sha256"],
        ),
        _validation_receipt(
            "independent_raw_row_reducer",
            ["in_process", "reduce_cost_rows"],
            0,
            canonical_json(summary),
        ),
        *external_receipts,
    ]
    artifact = _base_artifact(
        checks,
        hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - invocation_started,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": phase_spans,
            "rows": [*parity, *costs],
            "sample_size_budget": _sample_budget(True),
            "acceptance_gate_results": _acceptance_gates(summary),
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: exact parity, equal durability, and both interactive native cost gates passed"
                if value
                else "complete_null: complete event measurement found no accepted persistent-native interactive cost value"
            ),
            "validation_receipts": validation_receipts,
            "baseline_validation_failures": baseline_failures,
            "native_cost_complete_score": complete,
            "native_event_cost_value_score": value,
            "cost_rows": costs,
            "cost_summary": summary,
            "amortization_rows": amortization_rows(costs, setup),
            "native_binary_receipt": identity,
            "parity_rows": parity,
            "fresh_process_receipt": fresh,
            "independent_raw_row_reducer": summary,
            "checkpoint_receipt": checkpoint_receipt,
            "retirement_scope": (
                "no retirement; measured boundary cost gate passed"
                if value
                else "retire this boundary optimization only; Rust and sampler families remain active"
            ),
            "upstream_receipt": {
                "path": str(UPSTREAM_RELATIVE),
                "sha256": hashes[str(UPSTREAM_RELATIVE)],
                "native_controller_ready_score": evidence["upstream"].get(
                    "native_controller_ready_score"
                ),
                "historical_models_currently_invoked": False,
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, cold-validate, and atomically publish one terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / VALIDATION_RELATIVE,
        root / STORAGE_RELATIVE,
        output,
    )
    artifact = build_artifact(root, paths)
    progress(7, "before", "cold terminal artifact validation")
    errors = validate_artifact(
        artifact, check_files=artifact.get("status") == "complete", root=root
    )
    progress(7, "after", f"cold terminal artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7257 artifact:{errors}")
    progress(8, "before", "atomic terminal artifact write")
    receipt = atomic_write(output, artifact)
    progress(8, "after", f"atomic terminal artifact write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded measurement or validate existing bytes without mutation."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(7, "before", f"read-only validation path={args.validate}")
        try:
            artifact = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
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
