"""Diagnose durable cost in the shipped capacity-four archive controller.

The study first reduces every saved Exp7257 row. It then uses no more than
twelve local replay blocks to split the durable boundary. This keeps the old
speed headline intact while testing whether a new journal prototype has enough
replaceable work to matter.

Spec refs: REQ-RUSTPY-7270 and SCENARIO-RUSTPY-7270-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7270
SCHEMA = "carnot.exp7270.v639_durable_profile.v1"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_270_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
INSTRUMENTED_BLOCKS = 12
MAX_MEASUREMENT_S = 300.0

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7270_v639_durable_profile.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7270_v639_durable_profile.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7270_v639_saved_cost_rows.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7270_v639_terminal_candidate.json")
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7270_v639_durable_storage")
EXP7257_RELATIVE = Path("results/experiment_7257_v638_native_cost.json")
EXP7256_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/rust-python-boundary/spec.md")
EXPECTED_EXP7257_SHA256 = "sha256:2fd604d1ece4caa529908c2b4d71eb13fc664d1358372d21147515d6aca5ef1f"
EXPECTED_EXP7256_SHA256 = exp7257.EXPECTED_EXP7256_SHA256

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("python/carnot/experiment_7257_v638_native_cost.py"),
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("crates/carnot-python/src/experiment_7256_archive_controller.rs"),
    Path("scripts/experiments/experiment_7270_v639_durable_profile.py"),
    Path("tests/python/test_experiment_7270_v639_durable_profile.py"),
    SPEC_RELATIVE,
    EXP7257_RELATIVE,
    EXP7256_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted and completed model loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute and never pad time to satisfy a duration floor.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked work name the upstream, exact field, observed value and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements and blocked_* for external absence.",
    "verdict_class": "Use the closed verdict class; oracle evidence and failed scientific gates forbid positive.",
    "validation_receipts": "Record actual command, exit code and log hash; preserve every failure.",
    "durable_profile_complete_score": "One means measured decomposition and explicit unknowns are complete.",
    "journal_optimization_warranted_score": "One licenses only a cost-supported delta-log prototype; sync dominance keeps it zero.",
    "component_rows": "Per-block exclusive timings prevent double-counting an alleged bottleneck.",
    "acceleration_envelope": "State assumptions, unaccelerated fractions and attainable 10x and 100x bounds.",
    "replacement_cost_model": "Include fsync, amortized compaction and worst-case replay before claiming feasibility.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7257.canonical_json
sha256_file = exp7257.sha256_file
artifact_checksum = exp7257.artifact_checksum
check = exp7257.check
gate_summary = exp7257.gate_summary
atomic_write = exp7257.atomic_write
_finish_row = exp7257._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw, provisional, storage, and terminal bytes in separate paths."""

    checkpoint: Path
    raw_rows: Path
    raw_candidate: Path
    storage_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return paths used by the public experiment command."""

        return cls(
            REPO_ROOT / CHECKPOINT_RELATIVE,
            REPO_ROOT / RAW_ROWS_RELATIVE,
            REPO_ROOT / RAW_CANDIDATE_RELATIVE,
            REPO_ROOT / STORAGE_RELATIVE,
            REPO_ROOT / RESULT_RELATIVE,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test-owned evidence under one caller-owned directory."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / RAW_ROWS_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / STORAGE_RELATIVE,
            root / RESULT_RELATIVE,
        )

    def writable_targets(self) -> list[bool]:
        """Check ownership without creating a success-shaped output."""

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


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Print one flushed phase boundary with real monotonic elapsed time."""

    elapsed = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    print(f"[phase {phase} {boundary}] {operation}{elapsed}", flush=True)


def _writable(path: Path) -> bool:
    """Find the nearest existing parent and check its write permission."""

    parent = path if path.is_dir() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _read_json(path: Path) -> JsonDict:
    """Read one artifact and reject a non-object top level."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object:{path}")
    return value


def _checksum_valid(artifact: Mapping[str, Any]) -> bool:
    """Recompute an upstream checksum without trusting its declaration."""

    try:
        return artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        return False


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    exp7257_path: Path | None = None,
    exp7256_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate both source artifacts, native bytes, spec, and outputs."""

    started = time.monotonic()
    progress(0, "start", "authenticate Exp7257, Exp7256, spec, quarantine, and outputs", started)
    cost_path = exp7257_path or root / EXP7257_RELATIVE
    controller_path = exp7256_path or root / EXP7256_RELATIVE
    cost_artifact = _read_json(cost_path) if cost_path.is_file() else {}
    controller_artifact = _read_json(controller_path) if controller_path.is_file() else {}
    manifest = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    cost_quarantine = exp7213.quarantine_state(
        cost_artifact, manifest, cost_path.name, "exp7257-durable-cost"
    )
    controller_quarantine = exp7213.quarantine_state(
        controller_artifact, manifest, controller_path.name, "exp7256-native-controller"
    )
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    cost_hash = sha256_file(cost_path) if cost_path.is_file() else None
    controller_hash = sha256_file(controller_path) if controller_path.is_file() else None
    hashes[str(cost_path.resolve())] = cost_hash
    hashes[str(controller_path.resolve())] = controller_hash
    identity = controller_artifact.get("native_binary_receipt", {})
    module_path = Path(str(identity.get("module_file", "")))
    observed_module_hash = sha256_file(module_path) if module_path.is_file() else None
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
            "REQ-RUSTPY-7270",
            True,
            "REQ-RUSTPY-7270" in spec_text,
            "REQ-RUSTPY-7270" in spec_text,
        ),
        check(
            "exp7257_artifact_hash",
            str(cost_path),
            "sha256",
            EXPECTED_EXP7257_SHA256,
            cost_hash,
            cost_hash == EXPECTED_EXP7257_SHA256,
        ),
        check(
            "exp7257_complete_checksum",
            str(cost_path),
            "status,reproducibility_checksum,cost_rows",
            {"status": "complete", "checksum": True, "rows": 540},
            {
                "status": cost_artifact.get("status"),
                "checksum": _checksum_valid(cost_artifact),
                "rows": len(cost_artifact.get("cost_rows", [])),
            },
            cost_artifact.get("status") == "complete"
            and _checksum_valid(cost_artifact)
            and len(cost_artifact.get("cost_rows", [])) == 540,
        ),
        check(
            "exp7256_artifact_hash",
            str(controller_path),
            "sha256",
            EXPECTED_EXP7256_SHA256,
            controller_hash,
            controller_hash == EXPECTED_EXP7256_SHA256,
        ),
        check(
            "exp7256_ready_complete_checksum",
            str(controller_path),
            "status,native_controller_ready_score,reproducibility_checksum",
            {"status": "complete", "ready": 1, "checksum": True},
            {
                "status": controller_artifact.get("status"),
                "ready": controller_artifact.get("native_controller_ready_score"),
                "checksum": _checksum_valid(controller_artifact),
            },
            controller_artifact.get("status") == "complete"
            and controller_artifact.get("native_controller_ready_score") == 1
            and _checksum_valid(controller_artifact),
        ),
        check(
            "upstreams_not_quarantined",
            str(EXCLUSION_RELATIVE),
            "exp7257,exp7256 quarantined",
            {"exp7257": False, "exp7256": False},
            {
                "exp7257": cost_quarantine.get("quarantined"),
                "exp7256": controller_quarantine.get("quarantined"),
            },
            cost_quarantine.get("quarantined") is False
            and controller_quarantine.get("quarantined") is False,
        ),
        check(
            "native_binary_and_outputs",
            str(module_path),
            "module_sha256,writable outputs",
            {"binary": identity.get("module_sha256"), "outputs": [True] * 5},
            {"binary": observed_module_hash, "outputs": paths.writable_targets()},
            bool(identity.get("module_sha256"))
            and observed_module_hash == identity.get("module_sha256")
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
        "exp7257": cost_artifact,
        "exp7256": controller_artifact,
        "quarantine": {"exp7257": cost_quarantine, "exp7256": controller_quarantine},
        "native_module_path": str(module_path),
        "native_module_sha256": observed_module_hash,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic linear quantile for a nonempty sequence."""

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def reduce_saved_cost_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Cold-reduce all 540 saved rows without deleting durable outliers."""

    expected = (
        len(exp7257.ARCHIVE_CAPACITIES)
        * len(exp7257.BATCH_SIZES)
        * exp7257.PAIRED_BLOCKS
        * len(exp7257.ARMS)
    )
    keys = {
        (
            row.get("archive_capacity"),
            row.get("batch_size"),
            row.get("block"),
            row.get("arm"),
        )
        for row in rows
    }
    expected_keys = {
        (capacity, batch, block, arm)
        for capacity in exp7257.ARCHIVE_CAPACITIES
        for batch in exp7257.BATCH_SIZES
        for block in range(exp7257.PAIRED_BLOCKS)
        for arm in exp7257.ARMS
    }
    if len(rows) != expected or keys != expected_keys:
        raise ValueError(f"incomplete saved row roster:{len(rows)}/{expected}")

    component_profiles: list[JsonDict] = []
    sum_failures = 0
    for capacity in exp7257.ARCHIVE_CAPACITIES:
        for batch in exp7257.BATCH_SIZES:
            for arm in exp7257.ARMS:
                selected = [
                    row
                    for row in rows
                    if row["archive_capacity"] == capacity
                    and row["batch_size"] == batch
                    and row["arm"] == arm
                ]
                totals = [float(row["total_block_ns"]) for row in selected]
                component_names = tuple(sorted(selected[0]["component_ns"]))
                component_totals = {
                    name: sum(float(row["component_ns"][name]) for row in selected)
                    for name in component_names
                }
                total_sum = sum(totals)
                sums_match = all(
                    sum(float(value) for value in row["component_ns"].values())
                    == float(row["total_block_ns"])
                    for row in selected
                )
                sum_failures += int(not sums_match)
                component_profiles.append(
                    {
                        "archive_capacity": capacity,
                        "batch_size": batch,
                        "arm": arm,
                        "block_count": len(selected),
                        "censored_block_count": sum(bool(row.get("censored")) for row in selected),
                        "p50_total_block_ns": statistics.median(totals),
                        "p95_total_block_ns": _quantile(totals, 0.95),
                        "component_total_ns": component_totals,
                        "component_p50_ns": {
                            name: statistics.median(
                                float(row["component_ns"][name]) for row in selected
                            )
                            for name in component_names
                        },
                        "component_p95_ns": {
                            name: _quantile(
                                [float(row["component_ns"][name]) for row in selected], 0.95
                            )
                            for name in component_names
                        },
                        "component_fraction_of_total": {
                            name: value / total_sum for name, value in component_totals.items()
                        },
                        "unaccounted_total_ns": component_totals["residual_unaccounted_ns"],
                        "component_sum_matches_total": sums_match,
                        "storage_devices": sorted({int(row["storage_device"]) for row in selected}),
                        "arm_order_by_block": [
                            {"block": int(row["block"]), "order": int(row["arm_order"])}
                            for row in sorted(selected, key=lambda item: int(item["block"]))
                        ],
                    }
                )

    paired: list[JsonDict] = []
    for capacity in exp7257.ARCHIVE_CAPACITIES:
        for batch in exp7257.BATCH_SIZES:
            for comparison in exp7257.COMPARISON_ARMS:
                ratios: list[float] = []
                numerator_total = 0.0
                denominator_total = 0.0
                pair_keys: list[JsonDict] = []
                for block in range(exp7257.PAIRED_BLOCKS):
                    selected = [
                        row
                        for row in rows
                        if row["archive_capacity"] == capacity
                        and row["batch_size"] == batch
                        and row["block"] == block
                    ]
                    arms = {str(row["arm"]): row for row in selected}
                    numerator = float(arms["python_reference"]["total_event_ns"])
                    denominator = float(arms[comparison]["total_event_ns"])
                    ratios.append(numerator / denominator)
                    numerator_total += numerator
                    denominator_total += denominator
                    pair_keys.append(
                        {
                            "seed": int(arms["python_reference"]["seed"]),
                            "block": block,
                            "trace_sha256": arms["python_reference"]["trace_sha256"],
                        }
                    )
                mean_log = statistics.fmean(math.log(value) for value in ratios)
                paired.append(
                    {
                        "archive_capacity": capacity,
                        "batch_size": batch,
                        "comparison": f"python_reference_over_{comparison}",
                        "paired_block_count": len(ratios),
                        "pair_keys": pair_keys,
                        "primary_arithmetic_mean_of_paired_ratios": statistics.fmean(ratios),
                        "primary_matches_exp7257_headline_method": True,
                        "secondary_ratio_of_total_time": numerator_total / denominator_total,
                        "secondary_mean_paired_log_ratio": mean_log,
                        "secondary_geometric_mean_ratio": math.exp(mean_log),
                        "p50_paired_ratio": statistics.median(ratios),
                        "p95_paired_ratio": _quantile(ratios, 0.95),
                        "outlier_rows_removed": 0,
                    }
                )
    return {
        "inference_substrate": REDUCER_SUBSTRATE,
        "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "saved_row_count": len(rows),
        "expected_saved_row_count": expected,
        "censored_block_arm_count": sum(bool(row.get("censored")) for row in rows),
        "component_sum_failure_count": sum_failures,
        "component_profiles": component_profiles,
        "paired_ratio_diagnostics": paired,
        "outlier_policy": "retain every saved row, including durable tails",
    }


def _operation_accounting(state: Mapping[str, Any], release: Mapping[str, Any]) -> JsonDict:
    """Count source-level mask work on the authenticated fixed native path.

    The count covers comparisons, shifts, Boolean mask operations, and popcount
    calls whose loop lengths are visible in the Rust source. It excludes hidden
    instructions inside JSON, SHA-256, Python, and the operating system.
    """

    family = str(release["family_id"])
    value = int(release["numeric_value"]) % 33
    family_index = ("lower_bound", "upper_bound", "modular_equals", "cyclic_window").index(family)

    def accepted_count(index: int, raw_value: int) -> int:
        return sum(
            (
                raw_value >= parameter
                if index == 0
                else raw_value <= parameter
                if index == 1
                else raw_value == parameter
                if index == 2
                else (raw_value + 33 - parameter) % 33 < 8
            )
            for parameter in range(33)
        )

    full_vote_accepted = sum(
        accepted_count(index, raw_value) for index in range(4) for raw_value in range(33)
    )
    vote_passes = 3
    accept_mask_comparisons = vote_passes * 4 * 33 * 33
    vote_mask_calls = vote_passes * 4 * 33
    released_accepts = accepted_count(family_index, value)
    archive_count = len(state.get("archives", []))
    witness_count = min(len(state.get("release_window", [])) + 1, 16)
    nomination_calls = archive_count * witness_count
    comparisons = accept_mask_comparisons + 2 * 33 + nomination_calls * 33 + 12 + archive_count * 4
    bit_operations = (
        vote_passes * full_vote_accepted * 2
        + vote_mask_calls * 2
        + 2 * released_accepts * 2
        + nomination_calls * (2 * released_accepts + 2)
        + archive_count * 8
        + 10
    )
    return {
        "comparison_count": comparisons,
        "bit_operation_count": bit_operations,
        "scope": "source-level fixed-path operations; excludes serde, sha256, Python, and OS internals",
        "kan_guidance_only": True,
    }


def instrument_durable_blocks(
    binding: ModuleType,
    storage_dir: Path,
    *,
    blocks: int = INSTRUMENTED_BLOCKS,
    max_duration_s: float = MAX_MEASUREMENT_S,
) -> list[JsonDict]:
    """Split twelve or fewer persistent-controller durable event blocks."""

    if blocks < 1 or blocks > 12:
        raise ValueError("instrumented blocks must be between 1 and 12")
    storage_dir.mkdir(parents=True, exist_ok=True)
    durable_path = storage_dir / "durable_state.json"
    state = exp7257.seed_cost_state(4)
    initial_bytes = transactional.canonical_json_bytes(dict(state))
    started = time.monotonic()
    rows: list[JsonDict] = []
    last_report = started
    progress(4, "before", f"bounded durable benchmark planned_blocks={blocks}", started)
    for block in range(blocks):
        if time.monotonic() - started >= max_duration_s:
            break
        transactional._atomic_write(durable_path, initial_bytes)
        controller = exp7256.PersistentNativeArchiveController.from_state_with_binding(
            binding, state
        )
        release = exp7257._cost_release(block, 0)
        event = {
            "event_id": release["event_id"],
            "family_id": release["family_id"],
            "numeric_value": release["numeric_value"],
        }
        accounting = _operation_accounting(state, release)
        reference = exp7240.ArchivedBeliefController.from_state(state)
        expected_prediction = reference.predict(event)
        expected_energies = [reference.energy(label, event) for label in ("accept", "reject")]
        components: JsonDict = {}
        total_started = time.perf_counter_ns()

        part = time.perf_counter_ns()
        prediction = controller.predict(event)
        energies = [controller.energy(label, event) for label in ("accept", "reject")]
        components["lookup_ns"] = time.perf_counter_ns() - part
        part = time.perf_counter_ns()
        selected = controller.select_request([event], {str(event["event_id"]): 0})
        components["query_ns"] = time.perf_counter_ns() - part
        parent_hash = controller.state_hash()
        part = time.perf_counter_ns()
        receipt = controller.commit_batch(
            [release],
            current_cycle=int(release["release_index"]),
            expected_parent_hash=parent_hash,
        )
        components["update_ns"] = time.perf_counter_ns() - part

        part = time.perf_counter_ns()
        snapshot = controller.state_dict()
        components["snapshot_construction_ns"] = time.perf_counter_ns() - part
        part = time.perf_counter_ns()
        encoded = transactional.canonical_json_bytes(snapshot)
        components["encoding_ns"] = time.perf_counter_ns() - part
        part = time.perf_counter_ns()
        validated = exp7240.ArchivedBeliefController.from_state(snapshot)
        validated_bytes = validated.state_bytes()
        components["validation_ns"] = time.perf_counter_ns() - part

        part = time.perf_counter_ns()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{durable_path.name}.", suffix=".tmp", dir=storage_dir
        )
        temporary = Path(temporary_name)
        components["metadata_ns"] = time.perf_counter_ns() - part
        try:
            part = time.perf_counter_ns()
            written = 0
            while written < len(encoded):
                written += os.write(descriptor, encoded[written:])
            components["data_write_ns"] = time.perf_counter_ns() - part
            part = time.perf_counter_ns()
            os.fsync(descriptor)
            components["file_sync_ns"] = time.perf_counter_ns() - part
            os.close(descriptor)
            descriptor = -1
            part = time.perf_counter_ns()
            os.replace(temporary, durable_path)
            components["rename_ns"] = time.perf_counter_ns() - part
            part = time.perf_counter_ns()
            transactional._fsync_directory(storage_dir)
            components["directory_sync_ns"] = time.perf_counter_ns() - part
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary.exists():
                temporary.unlink()

        part = time.perf_counter_ns()
        durable_bytes = durable_path.read_bytes()
        components["parent_reload_ns"] = time.perf_counter_ns() - part
        part = time.perf_counter_ns()
        restored = exp7256.PersistentNativeArchiveController.from_snapshot(
            binding, durable_bytes.decode("utf-8")
        )
        restored_prediction = restored.predict(event)
        restored_hash = restored.state_hash()
        components["restore_ns"] = time.perf_counter_ns() - part
        total = time.perf_counter_ns() - total_started
        components["instrumentation_overhead_ns"] = max(
            total - sum(int(value) for value in components.values()), 0
        )
        state_hash = controller.state_hash()
        post_update_prediction = controller.predict(event)
        parity = (
            encoded == validated_bytes
            and state_hash == restored_hash
            and prediction == expected_prediction
            and energies == expected_energies
            and post_update_prediction == restored_prediction
            and selected["event_id"] == event["event_id"]
            and all(isinstance(energy, Mapping) for energy in energies)
            and receipt["new_state_hash"] == state_hash
        )
        durability = encoded == durable_bytes and durable_path.is_file()
        transfers = {
            "native_snapshot_to_python": len(canonical_json(snapshot).encode("utf-8")),
            "python_encoding": len(encoded),
            "data_write": len(encoded),
            "parent_reload": len(durable_bytes),
            "restore_input": len(durable_bytes),
        }
        row = _finish_row(
            {
                "unit_id": f"durable-profile:4:1:{block}:persistent_native_controller",
                "arm": "persistent_native_controller",
                "seed": RANDOM_SEED + block,
                "metric": total,
                "error": None,
                "abstention": False,
                "censored": False,
                "archive_capacity": 4,
                "batch_size": 1,
                "block": block,
                "total_event_ns": total,
                "exclusive_component_ns": components,
                "component_sum_matches_total": sum(components.values()) == total,
                "snapshot_bytes": len(encoded),
                "durable_bytes": len(durable_bytes),
                "transferred_bytes_by_boundary": transfers,
                "transferred_bytes": sum(transfers.values()),
                "comparison_count": accounting["comparison_count"],
                "bit_operation_count": accounting["bit_operation_count"],
                "operation_accounting_scope": accounting["scope"],
                "kan_cost_metrics_are_accounting_guidance_only": accounting["kan_guidance_only"],
                "storage_device": durable_path.stat().st_dev,
                "parity_passed": parity,
                "durability_passed": durability,
                "durability_receipt": {
                    "file_fsync": True,
                    "rename": True,
                    "directory_fsync": True,
                    "restore": True,
                },
                "parent_hash": parent_hash,
                "final_state_hash": state_hash,
                "restored_state_hash": restored_hash,
            }
        )
        rows.append(row)
        now = time.monotonic()
        if now - last_report >= 60:
            print(
                f"[phase 4 progress] completed_blocks={len(rows)}/{blocks} "
                f"elapsed_s={now - started:.3f}",
                flush=True,
            )
            last_report = now
    progress(4, "after", f"completed_blocks={len(rows)}/{blocks}", started)
    return rows


def synthetic_component_rows(blocks: int = 2, *, sync_ns: int = 60) -> list[JsonDict]:
    """Create compact exclusive rows for reducer and validator tests."""

    base = {
        "lookup_ns": 1,
        "query_ns": 1,
        "update_ns": 3,
        "snapshot_construction_ns": 8,
        "encoding_ns": 5,
        "validation_ns": 4,
        "metadata_ns": 4,
        "data_write_ns": 2,
        "file_sync_ns": sync_ns * 2 // 3,
        "rename_ns": 1,
        "directory_sync_ns": sync_ns - sync_ns * 2 // 3,
        "parent_reload_ns": 5,
        "restore_ns": 2,
        "instrumentation_overhead_ns": 4,
    }
    return [
        _finish_row(
            {
                "unit_id": f"fixture:{block}",
                "arm": "persistent_native_controller",
                "seed": RANDOM_SEED + block,
                "metric": 40 + sync_ns,
                "error": None,
                "abstention": False,
                "censored": False,
                "archive_capacity": 4,
                "batch_size": 1,
                "block": block,
                "total_event_ns": 40 + sync_ns,
                "exclusive_component_ns": dict(base),
                "component_sum_matches_total": True,
                "snapshot_bytes": 100,
                "durable_bytes": 100,
                "delta_log_entry_bytes": 10,
                "transferred_bytes": 500,
                "transferred_bytes_by_boundary": {"fixture": 500},
                "comparison_count": 100,
                "bit_operation_count": 1_000,
                "operation_accounting_scope": "fixture",
                "kan_cost_metrics_are_accounting_guidance_only": True,
                "storage_device": 1,
                "parity_passed": True,
                "durability_passed": True,
                "durability_receipt": {
                    "file_fsync": True,
                    "rename": True,
                    "directory_fsync": True,
                    "restore": True,
                },
                "parent_hash": "sha256:fixture-parent",
                "final_state_hash": "sha256:fixture-child",
                "restored_state_hash": "sha256:fixture-child",
            }
        )
        for block in range(blocks)
    ]


def reduce_component_rows(
    rows: Sequence[Mapping[str, Any]], *, planned_blocks: int = INSTRUMENTED_BLOCKS
) -> JsonDict:
    """Reduce exclusive component rows and expose every unknown or failure."""

    complete = [row for row in rows if row.get("censored") is not True]
    names = tuple(sorted(complete[0]["exclusive_component_ns"])) if complete else ()
    component_mean = {
        name: statistics.fmean(float(row["exclusive_component_ns"][name]) for row in complete)
        for name in names
    }
    total_mean = (
        statistics.fmean(float(row["total_event_ns"]) for row in complete) if complete else 0.0
    )
    sum_failures = sum(
        sum(float(value) for value in row["exclusive_component_ns"].values())
        != float(row["total_event_ns"])
        for row in complete
    )
    replaceable_names = (
        "snapshot_construction_ns",
        "parent_reload_ns",
        "encoding_ns",
        "metadata_ns",
    )
    sync_names = ("file_sync_ns", "directory_sync_ns")
    replaceable = sum(component_mean.get(name, 0.0) for name in replaceable_names)
    sync = sum(component_mean.get(name, 0.0) for name in sync_names)
    return {
        "planned_blocks": planned_blocks,
        "attempted_blocks": len(rows),
        "completed_blocks": len(complete),
        "censored_blocks": planned_blocks - len(complete),
        "component_names": list(names),
        "mean_total_event_ns": total_mean,
        "p50_total_event_ns": statistics.median(float(row["total_event_ns"]) for row in complete)
        if complete
        else None,
        "p95_total_event_ns": _quantile([float(row["total_event_ns"]) for row in complete], 0.95)
        if complete
        else None,
        "mean_component_ns": component_mean,
        "component_fraction_of_total": {
            name: value / total_mean for name, value in component_mean.items()
        }
        if total_mean
        else {},
        "replaceable_component_names": list(replaceable_names),
        "replaceable_snapshot_fraction": replaceable / total_mean if total_mean else 0.0,
        "fixed_sync_component_names": list(sync_names),
        "sync_fraction": sync / total_mean if total_mean else 0.0,
        "sync_dominates": sync / total_mean >= 0.5 if total_mean else False,
        "component_sum_failure_count": sum_failures,
        "parity_failure_count": sum(row.get("parity_passed") is not True for row in complete),
        "durability_failure_count": sum(
            row.get("durability_passed") is not True for row in complete
        ),
        "comparison_count": sum(int(row["comparison_count"]) for row in complete),
        "bit_operation_count": sum(int(row["bit_operation_count"]) for row in complete),
        "transferred_bytes": sum(int(row["transferred_bytes"]) for row in complete),
        "storage_devices": sorted({int(row["storage_device"]) for row in complete}),
    }


def acceleration_envelope(*, total_ns: float, fixed_ns: float) -> JsonDict:
    """Calculate Amdahl limits while the declared fixed work stays unchanged."""

    if total_ns <= 0 or fixed_ns < 0 or fixed_ns > total_ns:
        raise ValueError("invalid Amdahl costs")
    fixed_fraction = fixed_ns / total_ns
    replaceable_fraction = 1.0 - fixed_fraction
    targets: JsonDict = {}
    for target in (10, 100):
        denominator = 1.0 / target - fixed_fraction
        feasible = denominator > 0
        targets[f"{target}x"] = {
            "feasible": feasible,
            "required_replaceable_speedup": (total_ns - fixed_ns) / (total_ns / target - fixed_ns)
            if feasible
            else None,
        }
    return {
        "assumption": "validation, file sync, and directory sync remain fixed",
        "total_ns": total_ns,
        "fixed_ns": fixed_ns,
        "unaccelerated_fraction": fixed_fraction,
        "replaceable_fraction": replaceable_fraction,
        "best_case_speedup": math.inf if fixed_fraction == 0 else 1.0 / fixed_fraction,
        "targets": targets,
    }


def replacement_cost_model(
    component_summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Estimate a conservative delta log with sync, compaction, and replay."""

    means = component_summary["mean_component_ns"]
    snapshot_bytes = statistics.fmean(float(row["snapshot_bytes"]) for row in rows)
    entry_bytes = statistics.fmean(
        float(
            row.get(
                "delta_log_entry_bytes",
                len(canonical_json({"block": row["block"], "seed": row["seed"]}).encode()),
            )
        )
        for row in rows
    )
    byte_ratio = min(entry_bytes / snapshot_bytes, 1.0)
    sync = float(means["file_sync_ns"]) + float(means["directory_sync_ns"])
    fixed_core = sum(
        float(means[name])
        for name in ("lookup_ns", "query_ns", "update_ns", "validation_ns", "restore_ns")
    )
    scaled_encoding_and_write = (
        float(means["encoding_ns"]) + float(means["data_write_ns"])
    ) * byte_ratio
    per_commit_metadata = float(means["metadata_ns"]) + float(means["rename_ns"])
    compaction_interval = 16
    compaction = (
        sum(
            float(means[name])
            for name in (
                "snapshot_construction_ns",
                "encoding_ns",
                "metadata_ns",
                "data_write_ns",
                "file_sync_ns",
                "rename_ns",
                "directory_sync_ns",
            )
        )
        / compaction_interval
    )
    worst_case_replay = float(means["update_ns"]) * compaction_interval
    estimate = (
        fixed_core
        + sync
        + scaled_encoding_and_write
        + per_commit_metadata
        + compaction
        + worst_case_replay
    )
    current = float(component_summary["mean_total_event_ns"])
    return {
        "delta_log_entry_bytes": entry_bytes,
        "full_snapshot_bytes": snapshot_bytes,
        "entry_to_snapshot_byte_ratio": byte_ratio,
        "fixed_core_ns": fixed_core,
        "fixed_sync_ns": sync,
        "scaled_encoding_and_write_ns": scaled_encoding_and_write,
        "per_commit_metadata_and_rename_ns": per_commit_metadata,
        "compaction_interval_events": compaction_interval,
        "amortized_compaction_ns": compaction,
        "worst_case_replay_events": compaction_interval,
        "worst_case_replay_ns": worst_case_replay,
        "conservative_delta_log_event_ns": estimate,
        "conservative_delta_log_speedup": current / estimate,
    }


def journal_warrant(
    *,
    replaceable_fraction: float,
    conservative_delta_log_speedup: float,
    sync_fraction: float,
) -> JsonDict:
    """Apply both preregistered journal conditions and the sync veto."""

    sync_dominates = sync_fraction >= 0.5
    share_passed = replaceable_fraction >= 0.5
    speed_passed = conservative_delta_log_speedup >= 1.5
    score = int(share_passed and speed_passed and not sync_dominates)
    return {
        "replaceable_fraction_expected": ">=0.5",
        "replaceable_fraction_observed": replaceable_fraction,
        "replaceable_fraction_passed": share_passed,
        "delta_log_speedup_expected": ">=1.5",
        "delta_log_speedup_observed": conservative_delta_log_speedup,
        "delta_log_speedup_passed": speed_passed,
        "sync_fraction": sync_fraction,
        "sync_dominates": sync_dominates,
        "journal_optimization_warranted_score": score,
    }


def validation_receipt(name: str, command: Sequence[str], exit_code: int, output: str) -> JsonDict:
    """Bind one actual validation command to its complete output digest."""

    return {
        "name": name,
        "command": list(command),
        "exit_code": exit_code,
        "log_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
    }


def _sample_budget(saved_count: int, instrumented_count: int, planned: int) -> JsonDict:
    """Record both the fixed saved roster and the bounded replay roster."""

    return {
        "planned_saved_block_arm_units": 540,
        "attempted_saved_block_arm_units": saved_count,
        "completed_saved_block_arm_units": saved_count,
        "censored_saved_block_arm_units": 540 - saved_count,
        "planned_instrumented_blocks": planned,
        "attempted_instrumented_blocks": instrumented_count,
        "completed_instrumented_blocks": instrumented_count,
        "censored_instrumented_blocks": planned - instrumented_count,
        "stopping_rule": "cold-reduce all 540 saved rows; then exactly twelve fixed capacity-four batch-one blocks unless 300 seconds expires",
        "measurement_timeout_s": MAX_MEASUREMENT_S,
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
        "invocation_counts": {
            "attempted_model_loads": 0,
            "completed_model_loads": 0,
            "attempted_generation_calls": 0,
            "completed_generation_calls": 0,
            "usable_answers": 0,
        },
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
        "sample_size_budget": _sample_budget(0, 0, INSTRUMENTED_BLOCKS),
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external:unknown_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "durable_profile_complete_score": 0,
        "journal_optimization_warranted_score": 0,
        "component_rows": [],
        "acceleration_envelope": {},
        "replacement_cost_model": {},
        "saved_cost_rows": [],
        "saved_cost_reduction": {},
        "independent_raw_row_reducer": {},
        "component_reduction": {},
        "journal_warrant": {},
        "declared_bottleneck": None,
        "raw_rows_receipt": {"path": str(paths.raw_rows), "sha256": None},
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None},
        "methodology": {
            "saved_reducer": "read-only cold aggregation of authenticated Exp7257 rows",
            "bounded_replay": "CPU exact controller replay with explicit durable operation timers",
            "model_use": "no model load, generation, or inference",
            "device_claim": "none",
        },
        "retired_ownership_optimization_promoted": False,
        "device_speed_claimed": False,
        "publication_performed": False,
        "default_pipeline_modified": False,
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return one schema-complete row-free external block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["honest_verdict"] = f"blocked_external:{failed.get('check')}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _acceptance_gates(
    saved: Mapping[str, Any],
    components: Mapping[str, Any],
    envelope: Mapping[str, Any],
    warrant: Mapping[str, Any],
    *,
    planned_blocks: int = INSTRUMENTED_BLOCKS,
) -> JsonDict:
    """Keep completion, parity, feasibility, and prototype value distinct."""

    return {
        "complete_saved_roster": {
            "principle": "Every saved row must remain in the diagnostic.",
            "expected": 540,
            "observed": saved["saved_row_count"],
            "passed": saved["saved_row_count"] == 540,
        },
        "complete_bounded_profile": {
            "principle": "All fixed replay blocks must have exclusive component sums.",
            "expected": planned_blocks,
            "observed": components["completed_blocks"],
            "passed": components["completed_blocks"] == planned_blocks
            and components["component_sum_failure_count"] == 0,
        },
        "parity_and_durability": {
            "principle": "A cost diagnosis cannot weaken state or durability semantics.",
            "expected": 0,
            "observed": components["parity_failure_count"] + components["durability_failure_count"],
            "passed": components["parity_failure_count"] == 0
            and components["durability_failure_count"] == 0,
        },
        "journal_prototype_warrant": {
            "principle": "Both cost gates must pass and sync dominance vetoes the proposal.",
            "expected": 1,
            "observed": warrant["journal_optimization_warranted_score"],
            "passed": warrant["journal_optimization_warranted_score"] == 1,
        },
        "amdahl_10x_feasible": {
            "principle": "Fixed validation and sync work must permit a 10x limit.",
            "expected": True,
            "observed": envelope["targets"]["10x"]["feasible"],
            "passed": envelope["targets"]["10x"]["feasible"] is True,
        },
        "amdahl_100x_feasible": {
            "principle": "Fixed validation and sync work must permit a 100x limit.",
            "expected": True,
            "observed": envelope["targets"]["100x"]["feasible"],
            "passed": envelope["targets"]["100x"]["feasible"] is True,
        },
    }


def _assemble_complete_artifact(
    base: JsonDict,
    saved_rows: list[JsonDict],
    component_rows: list[JsonDict],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    raw_rows_receipt: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    planned_blocks: int = INSTRUMENTED_BLOCKS,
) -> JsonDict:
    """Join independent reductions and classify one complete diagnostic."""

    saved = reduce_saved_cost_rows(saved_rows)
    components = reduce_component_rows(component_rows, planned_blocks=planned_blocks)
    fixed = (
        float(components["mean_component_ns"]["validation_ns"])
        + float(components["mean_component_ns"]["file_sync_ns"])
        + float(components["mean_component_ns"]["directory_sync_ns"])
    )
    envelope = acceleration_envelope(total_ns=components["mean_total_event_ns"], fixed_ns=fixed)
    replacement = replacement_cost_model(components, component_rows)
    warrant = journal_warrant(
        replaceable_fraction=components["replaceable_snapshot_fraction"],
        conservative_delta_log_speedup=replacement["conservative_delta_log_speedup"],
        sync_fraction=components["sync_fraction"],
    )
    complete = int(
        saved["saved_row_count"] == 540
        and saved["component_sum_failure_count"] == 0
        and components["completed_blocks"] == planned_blocks
        and components["component_sum_failure_count"] == 0
        and components["parity_failure_count"] == 0
        and components["durability_failure_count"] == 0
    )
    journal_score = int(warrant["journal_optimization_warranted_score"])
    bottleneck = (
        "durable_sync"
        if components["sync_dominates"]
        else max(
            components["mean_component_ns"],
            key=lambda name: components["mean_component_ns"][name],
        )
    )
    base.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_substrates": {
                "saved_raw_reducer": {
                    "inference_substrate": REDUCER_SUBSTRATE,
                    "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
                },
                "bounded_replay": {
                    "inference_substrate": INFERENCE_SUBSTRATE,
                    "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                },
            },
            "rows": [*saved_rows, *component_rows],
            "sample_size_budget": _sample_budget(
                len(saved_rows), len(component_rows), planned_blocks
            ),
            "acceptance_gate_results": _acceptance_gates(
                saved, components, envelope, warrant, planned_blocks=planned_blocks
            ),
            "verdict_class": "circular_positive" if journal_score else "null",
            "honest_verdict": (
                "complete_circular_positive: cost evidence warrants only a bounded delta-log prototype"
                if journal_score
                else "complete_null: durable sync or insufficient replaceable snapshot work blocks a delta-log prototype"
            ),
            "validation_receipts": [dict(row) for row in validation_receipts],
            "durable_profile_complete_score": complete,
            "journal_optimization_warranted_score": journal_score,
            "component_rows": component_rows,
            "acceleration_envelope": envelope,
            "replacement_cost_model": replacement,
            "saved_cost_rows": saved_rows,
            "saved_cost_reduction": saved,
            "independent_raw_row_reducer": saved,
            "component_reduction": components,
            "journal_warrant": warrant,
            "declared_bottleneck": bottleneck,
            "raw_rows_receipt": dict(raw_rows_receipt),
            "checkpoint_receipt": dict(checkpoint_receipt),
        }
    )
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def complete_artifact_fixture_for_test() -> JsonDict:
    """Create one complete sync-dominated validator fixture."""

    saved_rows = exp7257.synthetic_cost_rows(python_ns=100, old_native_ns=90, persistent_ns=120)
    component_rows = synthetic_component_rows(INSTRUMENTED_BLOCKS)
    passed = check("fixture", "fixture", "fixture", True, True, True)
    now = datetime.now(UTC).isoformat()
    base = _base_artifact(
        [passed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    return _assemble_complete_artifact(
        base,
        saved_rows,
        component_rows,
        [validation_receipt("fixture", ["fixture"], 0, "ok")],
        raw_rows_receipt={"path": "fixture", "sha256": "sha256:fixture"},
        checkpoint_receipt={"path": "fixture", "sha256": "sha256:fixture"},
    )


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Recompute one row digest without trusting its stored value."""

    material = dict(row)
    stored = material.pop("row_sha256", None)
    return stored == "sha256:" + hashlib.sha256(canonical_json(material).encode()).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, raw reducers, component sums, gates, and verdict."""

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
        add(bool(artifact.get("rows")) or bool(artifact.get("component_rows")), "blocked_rows")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        return errors

    saved_rows = artifact.get("saved_cost_rows", [])
    component_rows = artifact.get("component_rows", [])
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    add(
        not isinstance(saved_rows, list)
        or not isinstance(component_rows, list)
        or artifact.get("rows") != [*saved_rows, *component_rows]
        or any(not _row_hash_valid(row) for row in [*saved_rows, *component_rows]),
        "rows",
    )
    try:
        saved = reduce_saved_cost_rows(saved_rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        saved = None
    add(
        saved is None
        or artifact.get("saved_cost_reduction") != saved
        or artifact.get("independent_raw_row_reducer") != saved,
        "saved_reducer",
    )
    planned = int(artifact.get("sample_size_budget", {}).get("planned_instrumented_blocks", 0))
    try:
        components = reduce_component_rows(component_rows, planned_blocks=planned)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        components = None
    add(
        components is None or artifact.get("component_reduction") != components, "component_reducer"
    )
    if saved is not None and components is not None and components["mean_total_event_ns"] > 0:
        means = components["mean_component_ns"]
        fixed = (
            float(means["validation_ns"])
            + float(means["file_sync_ns"])
            + float(means["directory_sync_ns"])
        )
        envelope = acceleration_envelope(total_ns=components["mean_total_event_ns"], fixed_ns=fixed)
        replacement = replacement_cost_model(components, component_rows)
        warrant = journal_warrant(
            replaceable_fraction=components["replaceable_snapshot_fraction"],
            conservative_delta_log_speedup=replacement["conservative_delta_log_speedup"],
            sync_fraction=components["sync_fraction"],
        )
        add(artifact.get("acceleration_envelope") != envelope, "acceleration_envelope")
        add(artifact.get("replacement_cost_model") != replacement, "replacement_cost_model")
        add(
            artifact.get("journal_warrant") != warrant
            or artifact.get("journal_optimization_warranted_score")
            != warrant["journal_optimization_warranted_score"],
            "journal_warrant",
        )
        complete = int(
            saved["saved_row_count"] == 540
            and saved["component_sum_failure_count"] == 0
            and components["completed_blocks"] == planned == INSTRUMENTED_BLOCKS
            and components["component_sum_failure_count"] == 0
            and components["parity_failure_count"] == 0
            and components["durability_failure_count"] == 0
        )
        add(artifact.get("durable_profile_complete_score") != complete, "complete_score")
        add(
            artifact.get("acceptance_gate_results")
            != _acceptance_gates(saved, components, envelope, warrant, planned_blocks=planned),
            "acceptance_gate_results",
        )
        expected_class = (
            "circular_positive" if warrant["journal_optimization_warranted_score"] == 1 else "null"
        )
        add(
            artifact.get("verdict_class") != expected_class
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
    """Reuse the shipped unbuffered stream and preserve a failed exit receipt."""

    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{root / 'python'}:{root}",
    }
    try:
        return exp7217._stream_process(
            command,
            root=root,
            environment=environment,
            operation=operation,
            timeout_s=timeout_s,
        )
    except RuntimeError as error:
        return {"command": command, "exit_code": 1, "output": str(error)}


def _scoped_validation_commands(root: Path) -> list[tuple[str, list[str], int]]:
    """Return the exact focused test, coverage, lint, type, and spec checks."""

    python = str(Path(sys.executable).absolute())
    new_test = "tests/python/test_experiment_7270_v639_durable_profile.py"
    changed = [
        "python/carnot/experiment_7270_v639_durable_profile.py",
        "scripts/experiments/experiment_7270_v639_durable_profile.py",
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
                "tests/python/test_experiment_7257_v638_native_cost.py",
                "tests/python/test_experiment_7256_v638_native_controller.py",
                "-q",
                "-n",
                "0",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7270-affected",
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
                    "cov=coverage.Coverage(source=['carnot.experiment_7270_v639_durable_profile']); "
                    "cov.erase(); cov.start(); import pytest; "
                    f"rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','{new_test}',"
                    "'-q','-n','0','--basetemp=/tmp/carnot-exp7270-coverage']); "
                    "cov.stop(); cov.save(); percent=cov.report(file=sys.stdout,show_missing=True); "
                    "raise SystemExit(rc if rc else (0 if percent >= 100.0 else 1))"
                ),
            ],
            600,
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *changed], 120),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *changed], 120),
        (
            "changed_module_mypy",
            [python, "-m", "mypy", changed[0], changed[1]],
            300,
        ),
        ("spec_coverage", [python, "scripts/check_spec_coverage.py", new_test], 120),
    ]


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    instrumented_blocks: int = INSTRUMENTED_BLOCKS,
    precondition_bundle: tuple[list[JsonDict], JsonDict] | None = None,
) -> JsonDict:
    """Reduce saved evidence, run bounded replay, and build one candidate."""

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
        base["honest_verdict"] = "blocked_external:" + str(
            base["gate_check_summary"]["failed_check"]
        )
        base["reproducibility_checksum"] = artifact_checksum(base)
        return base

    progress(1, "start", "MODEL_SPECS empty; model loads=0 generations=0", invocation_started)
    progress(1, "end", "no model call authorized or attempted", invocation_started)
    phase_spans: JsonDict = {}
    phase = time.monotonic()
    cost_artifact = _read_json(root / EXP7257_RELATIVE)
    saved_rows = [dict(row) for row in cost_artifact["cost_rows"]]
    saved = reduce_saved_cost_rows(saved_rows)
    raw_rows_receipt = atomic_write(
        paths.raw_rows,
        {
            "schema": "carnot.exp7270.saved_cost_rows.v1",
            "source": str(EXP7257_RELATIVE),
            "source_sha256": evidence.get("hashes", {}).get(str(EXP7257_RELATIVE)),
            "inference_substrate": REDUCER_SUBSTRATE,
            "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
            "rows": saved_rows,
            "reduction": saved,
        },
    )
    phase_spans["cold_saved_row_reduction"] = time.monotonic() - phase

    progress(3, "before", "load authenticated shipped Exp7256 native extension", invocation_started)
    native_path = (
        evidence.get("native_module_path")
        or evidence["exp7256"]["native_binary_receipt"]["module_file"]
    )
    binding = exp7230.load_native_extension(Path(native_path))
    progress(3, "after", "authenticated native extension loaded", invocation_started)
    phase = time.monotonic()
    component_rows = instrument_durable_blocks(
        binding,
        paths.storage_dir,
        blocks=instrumented_blocks,
        max_duration_s=MAX_MEASUREMENT_S,
    )
    phase_spans["bounded_instrumented_replay"] = time.monotonic() - phase
    checkpoint_receipt = atomic_write(
        paths.checkpoint,
        {
            "schema": "carnot.exp7270.checkpoint.v1",
            "status": "provisional_measurement_complete"
            if len(component_rows) == instrumented_blocks
            else "partial_measurement_retryable",
            "component_rows": component_rows,
            "component_reduction": reduce_component_rows(
                component_rows, planned_blocks=instrumented_blocks
            ),
        },
    )
    if len(component_rows) != instrumented_blocks:
        raise RuntimeError("bounded replay incomplete; retry from checkpoint")
    base["completed_at_utc"] = datetime.now(UTC).isoformat()
    base["duration_s"] = time.monotonic() - invocation_started
    base["phase_spans_s"] = phase_spans
    artifact = _assemble_complete_artifact(
        base,
        saved_rows,
        component_rows,
        validation_receipts,
        raw_rows_receipt=raw_rows_receipt,
        checkpoint_receipt=checkpoint_receipt,
        planned_blocks=instrumented_blocks,
    )
    return artifact


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Validate, measure, cold-check, then atomically publish terminal bytes."""

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
            root,
            paths,
            validation_receipts=[],
            precondition_bundle=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked Exp7270 artifact:{errors}")
        atomic_write(output, artifact)
        return artifact

    receipts: list[JsonDict] = []
    for name, command, timeout_s in _scoped_validation_commands(root):
        progress(2, "before", f"validation subprocess {name}")
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        progress(2, "after", f"validation subprocess {name} exit_code={result['exit_code']}")
        receipts.append(
            validation_receipt(
                name, command, int(result["exit_code"]), str(result.get("output", ""))
            )
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7270.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"scoped validation failed:{name}")

    artifact = build_artifact(
        root,
        paths,
        validation_receipts=receipts,
        precondition_bundle=preconditions,
    )
    progress(5, "before", "cold in-process candidate validation")
    errors = validate_artifact(artifact)
    progress(5, "after", f"cold in-process candidate validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7270 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7270_v639_durable_profile.py",
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
    for index, command in enumerate(commands, start=1):
        name = f"terminal_candidate_check_{index}"
        progress(6, "before", name)
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        progress(6, "after", f"{name} exit_code={result['exit_code']}")
        if result["exit_code"] != 0:
            raise RuntimeError(f"candidate validation failed:{name}")
    progress(7, "before", "atomic terminal artifact publication")
    atomic_write(output, artifact)
    progress(7, "after", f"atomic terminal artifact publication path={output}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date, output, and read-only validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded profile or validate existing bytes without mutation."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(5, "before", f"read-only validation path={args.validate}")
        try:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(5, "after", f"read-only validation errors={len(errors)}")
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
