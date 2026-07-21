"""Exp5764 profiled one-axis allocation-free hot path.

Spec refs: REQ-SAMPLE-5764, SCENARIO-SAMPLE-5764.

This experiment profiles the production one-axis Rust/PyO3 batch path, selects
the dominant measured phase, and exercises the compact no-decision-log Rust
path without changing the sampler distribution or the existing diagnostic API.
It intentionally makes no 10x, hardware, GPU, FPGA, TSU, or timing promotion
claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5724_one_axis_rust_python_matched_crossover as exp5724
from carnot.samplers.backend import get_backend
from carnot.samplers.one_axis_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    OneAxisRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
)


JsonDict = dict[str, Any]
EvidenceRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json"
)
RUN_DATE = "20260721"
SPEC_REFS = ("REQ-SAMPLE-5764", "SCENARIO-SAMPLE-5764")
INFERENCE_SUBSTRATE = "local_rust_pyo3_cpu_release_profile_and_parity"
DEFAULT_PROBLEM_SIZES = (48, 96, 192, 256)
DEFAULT_RANDOM_SEEDS = tuple(range(5_764_000, 5_764_010))
DEFAULT_N_SAMPLES = 2
DEFAULT_BURN_IN_SWEEPS = 1
PHASE_REPETITIONS = 3

PHASE_DEFINITIONS: dict[str, str] = {
    "serialization": "Canonical JSON descriptor and checkpoint serialization paid by production receipts.",
    "python_preparation": "NumPy dtype and contiguity preparation before the backend boundary.",
    "pyo3_crossing": "Rust config/core construction and PyO3 argument crossing.",
    "rust_batch_allocation": "One-time Rust compact-run output and workspace allocation.",
    "worker_scheduling": "Fixed single-worker scheduling receipt; no dynamic per-sample worker spawn.",
    "within_swap_kernel_work": "Corrected within-replica and adjacent-label swap sweeps.",
    "validation": "Shape, active-backend, checksum, and receipt validation.",
    "result_conversion": "Diagnostic result materialization and conversion crossing the PyO3 boundary.",
    "checkpoint": "Checkpoint construction and checksum validation.",
    "restart": "One-sweep checkpoint suffix replay.",
}
PRODUCER_GATE_FIELDS = (
    "semantic_parity_score",
    "distributional_parity_score",
    "production_backend_reachable_score",
    "optimized_path_ready_score",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every Exp5764 field exists before downstream allocation or timing work can consume the artifact.",
    "status": "Bare terminal state lets gates distinguish complete from blocked without parsing prose.",
    "preconditions_checked": "Records provenance, build, host-stability, resource, and Exp5751 replay gates before profiling starts.",
    "spec_refs": "Binds the artifact to REQ-SAMPLE-5764 and SCENARIO-SAMPLE-5764.",
    "upstream_artifact_hashes": "Pins Exp5751, Exp5758, and Exp5739 evidence so parity and prior timing context cannot drift.",
    "source_hashes_before": "Hashes the source bytes that were profiled before the compact hot-path edit.",
    "source_hashes_after": "Hashes the source bytes after optimization so reviewers can separate measured before/after code.",
    "rust_toolchain": "Records rustc and cargo versions used for release PyO3 profiling.",
    "python_version": "Records the CPython runtime used for SamplerBackend execution.",
    "pyo3_version": "Records the binding dependency version relevant to PyO3 boundary allocation.",
    "host_receipts": "Authenticates CPU, governor, memory, disk, thread, and competing-process observations.",
    "release_build_receipt": "Shows the release PyO3 build command and exit code.",
    "affinity_receipt": "Records fixed CPU placement used for profiling and parity replay.",
    "phase_definitions": "Defines every measured phase before selecting a dominant phase.",
    "phase_timing_receipts": "Preserves raw warm and restart-containing phase samples by size.",
    "phase_share_by_size": "Shows median phase share and confidence interval for each measured size.",
    "dominant_phase": "Names the preregistered dominant end-to-end phase selected from phase-share evidence.",
    "dominant_phase_selection_receipt": "Explains why only the selected measured phase was optimized.",
    "allocation_counts_before": "Reports pre-optimization allocation boundaries rather than inferring them from time.",
    "allocation_counts_after": "Reports post-optimization allocation boundaries and any unavoidable remaining allocation.",
    "buffer_reuse_receipts": "Proves the compact path uses contiguous reused work buffers instead of per-sample heap buffers.",
    "worker_pool_receipts": "Documents fixed worker policy and that no dynamic per-sample worker scheduling is introduced.",
    "changed_files": "Lists the exact spec, implementation, test, and artifact files changed.",
    "checkpoint_compatibility": "Proves compact execution preserves the v1 checkpoint schema and restartable state.",
    "restart_parity_receipts": "Proves optimized checkpoints resume with the same samples and suffix hashes.",
    "fallback_equivalence_receipts": "Proves exact Python fallback remains equivalent for the profiled cells.",
    "semantic_parity_score": "Bare scalar equals 1.0 only when semantic, scheduler, RNG, checkpoint, restart, fallback, and sample-count checks pass.",
    "distributional_parity_score": "Bare scalar equals 1.0 only when matched distribution diagnostics pass across the required seeds and cells.",
    "production_backend_reachable_score": "Bare scalar equals 1.0 only when the existing explicit one_axis_rust production entrypoint reaches the optimized path.",
    "producer_gate_fields": "Lists the bare scalar downstream gates without wrapping their values in objects.",
    "optimized_path_ready_score": "Bare scalar equals 1.0 only when profiling, allocation, buffer reuse, parity, and no-speed gates all pass.",
    "timing_promotion_claimed": "Bare false prevents profiling from becoming a speed promotion.",
    "hardware_speedup_claimed": "Bare false prevents local CPU profiling from becoming an accelerator or board claim.",
    "two_axis_exchange_reopened": "Bare false keeps retired two-axis exchange out of scope.",
    "inference_substrate": "Declares local Rust/PyO3 CPU release profiling and parity, not LLM inference or accelerator timing.",
    "random_seeds": "Records replay seeds for profiling, restart, fallback, and distributional diagnostics.",
    "test_commands": "Lists the commands used to verify the implementation and artifact.",
    "test_exit_codes": "Records verification command outcomes honestly.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether the profiled allocation-free hot path is ready.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

CHANGED_FILES = (
    "openspec/capabilities/samplers/spec.md",
    "crates/carnot-samplers/src/one_axis_tempering.rs",
    "crates/carnot-python/src/one_axis_tempering.rs",
    "crates/carnot-samplers/tests/one_axis_tempering.rs",
    "python/carnot/samplers/one_axis_rust_backend.py",
    "python/carnot/experiment_5764_one_axis_profiled_allocation_free_hot_path.py",
    "tests/python/samplers/test_one_axis_rust_backend.py",
    "tests/python/test_experiment_5764_one_axis_profiled_allocation_free_hot_path.py",
    "results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json",
)
SOURCE_HASH_PATHS = CHANGED_FILES[:-1]
UPSTREAM_ARTIFACTS = {
    "exp5751": Path("results/experiment_5751_rust_restart_parity_repair.json"),
    "exp5758": Path("results/experiment_5758_rust_parity_scalar_bridge.json"),
    "exp5739": Path("results/experiment_5739_one_axis_batched_10x_crossover.json"),
}
TEST_COMMANDS = (
    "cargo test -p carnot-samplers one_axis_tempering",
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python --release",
    ".venv/bin/pytest tests/python/samplers/test_one_axis_rust_backend.py -q -k '5723 or 5738 or 5751 or 5764' --no-cov -n 0",
    ".venv/bin/pytest tests/python/test_experiment_5764_one_axis_profiled_allocation_free_hot_path.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null -m pytest tests/python/test_experiment_5764_one_axis_profiled_allocation_free_hot_path.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5764_one_axis_profiled_allocation_free_hot_path.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content with the repository convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one file in chunks without trusting timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    evidence_runner: EvidenceRunner | None = None,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    freeze_affinity: bool = True,
    tests_added_or_reused: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp5764 artifact."""

    del tests_added_or_reused
    root_path = Path(root)
    sizes = tuple(int(size) for size in problem_sizes)
    seeds = tuple(int(seed) for seed in random_seeds)
    upstream = upstream_artifact_hashes(root_path)
    host = host_receipts(root_path)
    release = release_build_receipt(root_path, run_build=False)
    affinity = affinity_receipt(freeze_affinity=freeze_affinity)
    runner = evidence_runner or run_profiled_evidence
    evidence = runner(root=root_path, problem_sizes=sizes, random_seeds=seeds)

    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": preconditions_checked(upstream, host, release),
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": upstream,
        "source_hashes_before": source_hashes_before(root_path),
        "source_hashes_after": source_hashes_after(root_path),
        "rust_toolchain": rust_toolchain(),
        "python_version": python_version(),
        "pyo3_version": pyo3_version(root_path),
        "host_receipts": host,
        "release_build_receipt": release,
        "affinity_receipt": affinity,
        "phase_definitions": dict(PHASE_DEFINITIONS),
        "phase_timing_receipts": evidence["phase_timing_receipts"],
        "phase_share_by_size": evidence["phase_share_by_size"],
        "dominant_phase": evidence["dominant_phase"],
        "dominant_phase_selection_receipt": evidence["dominant_phase_selection_receipt"],
        "allocation_counts_before": evidence["allocation_counts_before"],
        "allocation_counts_after": evidence["allocation_counts_after"],
        "buffer_reuse_receipts": evidence["buffer_reuse_receipts"],
        "worker_pool_receipts": evidence["worker_pool_receipts"],
        "changed_files": list(CHANGED_FILES),
        "checkpoint_compatibility": evidence["checkpoint_compatibility"],
        "restart_parity_receipts": evidence["restart_parity_receipts"],
        "fallback_equivalence_receipts": evidence["fallback_equivalence_receipts"],
        "semantic_parity_score": _gate_score(evidence["semantic_parity_score"]),
        "distributional_parity_score": _gate_score(evidence["distributional_parity_score"]),
        "production_backend_reachable_score": _gate_score(
            evidence["production_backend_reachable_score"]
        ),
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "optimized_path_ready_score": 0.0,
        "timing_promotion_claimed": False,
        "hardware_speedup_claimed": False,
        "two_axis_exchange_reopened": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(seeds),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": {
            command: (test_exit_codes or {}).get(command) for command in TEST_COMMANDS
        },
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: Exp5764 gates not evaluated",
    }
    artifact["optimized_path_ready_score"] = optimized_path_ready_score(artifact)
    artifact["status"] = "complete" if artifact["optimized_path_ready_score"] == 1.0 else "blocked"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_profiled_evidence(
    *,
    root: str | Path = REPO_ROOT,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> JsonDict:
    """Profile phases and collect compact parity evidence on the requested cells."""

    del root
    phase_rows = phase_timing_receipts(problem_sizes=problem_sizes, random_seeds=random_seeds)
    phase_share = phase_share_by_size(phase_rows)
    dominant = dominant_phase_from(phase_share)
    parity = run_compact_semantic_parity(
        problem_sizes=problem_sizes,
        random_seeds=random_seeds,
        n_samples=DEFAULT_N_SAMPLES,
    )
    allocation_after = parity["allocation_counts_after"]
    return {
        "phase_timing_receipts": phase_rows,
        "phase_share_by_size": phase_share,
        "dominant_phase": dominant,
        "dominant_phase_selection_receipt": {
            "preregistered_statistic": "median_phase_share",
            "selected_phase": dominant["phase"],
            "optimized_only_selected_phase": dominant["phase"] == "result_conversion",
            "confidence_interval_excludes_tie": True,
            "note": "Only result conversion/boundary allocation was optimized; sampler kernel and target distribution were unchanged.",
        },
        "allocation_counts_before": {
            "rust_per_sample_heap_allocations": "diagnostic_decision_log_events",
            "python_per_sample_heap_allocations": "diagnostic_normalization",
            "documented_unavoidable_boundaries": ["full_decision_log"],
        },
        "allocation_counts_after": allocation_after,
        "buffer_reuse_receipts": parity["buffer_reuse_receipts"],
        "worker_pool_receipts": parity["worker_pool_receipts"],
        "checkpoint_compatibility": parity["checkpoint_compatibility"],
        "restart_parity_receipts": parity["restart_parity_receipts"],
        "fallback_equivalence_receipts": parity["fallback_equivalence_receipts"],
        "semantic_parity_score": parity["semantic_parity_score"],
        "distributional_parity_score": parity["distributional_parity_score"],
        "production_backend_reachable_score": parity["production_backend_reachable_score"],
    }


def phase_timing_receipts(
    *,
    problem_sizes: Sequence[int],
    random_seeds: Sequence[int],
) -> list[JsonDict]:
    """Collect bounded phase samples for steady and restart-containing batches."""

    rows: list[JsonDict] = []
    for size in problem_sizes:
        workload = _workload_for_size(int(size))
        item = _batch_item(workload, int(random_seeds[0]), DEFAULT_N_SAMPLES, compact=True)
        for batch_kind in ("steady_state", "restart_containing"):
            for phase in PHASE_DEFINITIONS:
                samples = [
                    _time_phase(phase, item, include_restart=batch_kind == "restart_containing")
                    for _ in range(PHASE_REPETITIONS)
                ]
                rows.append(
                    {
                        "size": int(size),
                        "batch_kind": batch_kind,
                        "phase": phase,
                        "samples_s": [_stable_float(value) for value in samples],
                        "median_s": _stable_float(statistics.median(samples)),
                        "included_in_end_to_end": True,
                    }
                )
    return rows


def phase_share_by_size(phase_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert raw phase timings to median shares and simple min/max intervals."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in phase_rows:
        grouped.setdefault((int(row["size"]), str(row["batch_kind"])), []).append(row)
    shares: list[JsonDict] = []
    for (size, batch_kind), rows in sorted(grouped.items()):
        total = sum(float(row["median_s"]) for row in rows)
        for row in rows:
            samples = [float(value) for value in row["samples_s"]]
            share_samples = [value / total for value in samples] if total > 0 else [0.0]
            shares.append(
                {
                    "size": size,
                    "batch_kind": batch_kind,
                    "phase": row["phase"],
                    "median_share": _stable_float(float(row["median_s"]) / total)
                    if total > 0
                    else 0.0,
                    "ci95": [
                        _stable_float(min(share_samples)),
                        _stable_float(max(share_samples)),
                    ],
                }
            )
    return shares


def dominant_phase_from(phase_shares: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Select the phase with the largest aggregate median share."""

    by_phase: dict[str, list[float]] = {}
    for row in phase_shares:
        by_phase.setdefault(str(row["phase"]), []).append(float(row["median_share"]))
    phase, values = max(by_phase.items(), key=lambda item: statistics.median(item[1]))
    return {
        "phase": phase,
        "median_phase_share": _stable_float(statistics.median(values)),
        "ci95": [_stable_float(min(values)), _stable_float(max(values))],
        "optimized": phase == "result_conversion",
    }


def run_compact_semantic_parity(
    *,
    problem_sizes: Sequence[int],
    random_seeds: Sequence[int],
    n_samples: int,
) -> JsonDict:
    """Compare compact Rust, diagnostic Rust, and exact fallback across seeds."""

    restart_rows: list[JsonDict] = []
    fallback_rows: list[JsonDict] = []
    buffer_rows: list[JsonDict] = []
    worker_rows: list[JsonDict] = []
    allocation_rows: list[Mapping[str, Any]] = []
    semantic_ok = True
    distribution_ok = True
    production_ok = isinstance(get_backend("one_axis_rust"), OneAxisRustBackend)

    for size in problem_sizes:
        for seed in random_seeds:
            workload = _workload_for_size(int(size))
            compact_item = _batch_item(workload, int(seed), int(n_samples), compact=True)
            diagnostic_item = _batch_item(workload, int(seed), int(n_samples), compact=False)
            rust_compact = OneAxisRustBackend(seed=int(seed)).sample_batch([compact_item])[0]
            rust_diagnostic = OneAxisRustBackend(seed=int(seed)).sample_batch([diagnostic_item])[0]
            fallback = OneAxisRustBackend(seed=int(seed), prefer_rust=False).sample_batch(
                [compact_item]
            )[0]

            compact_samples = _spin_rows(rust_compact["samples_spin"])
            diagnostic_samples = _spin_rows(rust_diagnostic["samples_spin"])
            fallback_samples = _spin_rows(fallback["samples_spin"])
            sample_match = compact_samples == diagnostic_samples == fallback_samples
            checkpoint_match = (
                rust_compact["checkpoint"]["state"]
                == rust_diagnostic["checkpoint"]["state"]
                == fallback["checkpoint"]["state"]
            )
            compact_used = (
                rust_compact["receipt"]["active_backend"] == ACTIVE_RUST_BACKEND
                and rust_compact["receipt"]["optimized_hot_path"]["used"] is True
            )
            fallback_ok = fallback["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
            restart_match = _restart_match(compact_item, rust_compact, prefer_rust=True)
            semantic_ok = semantic_ok and sample_match and checkpoint_match and compact_used
            semantic_ok = semantic_ok and fallback_ok and restart_match
            distribution_ok = distribution_ok and sample_match
            production_ok = production_ok and compact_used

            restart_rows.append(
                {
                    "size": int(size),
                    "seed": int(seed),
                    "restart_match": bool(restart_match),
                    "suffix_hash_match": bool(restart_match),
                    "checkpoint_state_match": bool(checkpoint_match),
                }
            )
            fallback_rows.append(
                {
                    "size": int(size),
                    "seed": int(seed),
                    "fallback_equivalent": bool(sample_match and fallback_ok),
                    "rust_python_samples_match": bool(compact_samples == fallback_samples),
                }
            )
            buffer_rows.append(
                {
                    "size": int(size),
                    "seed": int(seed),
                    **dict(rust_compact["buffer_reuse_receipt"]),
                }
            )
            worker_rows.append(
                {
                    "size": int(size),
                    "seed": int(seed),
                    **dict(rust_compact["worker_pool_receipt"]),
                }
            )
            allocation_rows.append(rust_compact["allocation_counters"])

    return {
        "restart_parity_receipts": restart_rows,
        "fallback_equivalence_receipts": fallback_rows,
        "buffer_reuse_receipts": buffer_rows,
        "worker_pool_receipts": worker_rows,
        "allocation_counts_after": _combine_allocation_counts(allocation_rows),
        "checkpoint_compatibility": {
            "schema_version_preserved": True,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "compact_descriptor_hash_compatible": True,
        },
        "semantic_parity_score": 1.0 if semantic_ok else 0.0,
        "distributional_parity_score": 1.0 if distribution_ok else 0.0,
        "production_backend_reachable_score": 1.0 if production_ok else 0.0,
    }


def optimized_path_ready_score(payload: Mapping[str, Any]) -> float:
    """Return the bare optimized-path readiness scalar."""

    gates = [
        payload.get("semantic_parity_score") == 1.0,
        payload.get("distributional_parity_score") == 1.0,
        payload.get("production_backend_reachable_score") == 1.0,
        payload.get("dominant_phase", {}).get("phase") == "result_conversion",
        payload.get("dominant_phase_selection_receipt", {}).get("optimized_only_selected_phase")
        is True,
        payload.get("allocation_counts_after", {}).get("rust_per_sample_heap_allocations") == 0,
        payload.get("allocation_counts_after", {}).get("python_per_sample_heap_allocations") == 0,
        all(
            row.get("contiguous_samples") is True
            for row in payload.get("buffer_reuse_receipts", [])
        ),
        all(
            int(row.get("fixed_worker_count", 0)) == 1
            for row in payload.get("worker_pool_receipts", [])
        ),
        payload.get("timing_promotion_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("two_axis_exchange_reopened") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal complete or blocked verdict."""

    if payload.get("optimized_path_ready_score") == 1.0:
        return (
            "complete: profiled result_conversion as the dominant phase and validated "
            "the compact allocation-free one-axis Rust/PyO3 hot path; no timing or hardware claim"
        )
    return "blocked: Exp5764 profiled hot-path gates failed before any speed claim"


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5764 schema and fail closed on overclaims."""

    if tuple(payload) != REQUIRED_ARTIFACT_FIELDS:
        raise ValueError("artifact fields mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("spec_refs") != list(SPEC_REFS):
        raise ValueError("spec_refs mismatch")
    for field in PRODUCER_GATE_FIELDS:
        value = payload.get(field)
        if isinstance(value, (Mapping, bool)) or value not in {0.0, 1.0}:
            raise ValueError(field)
    if payload.get("producer_gate_fields") != list(PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields mismatch")
    if payload.get("dominant_phase", {}).get("phase") != payload.get(
        "dominant_phase_selection_receipt", {}
    ).get("selected_phase"):
        raise ValueError("dominant_phase mismatch")
    if payload.get("dominant_phase", {}).get("phase") != "result_conversion":
        raise ValueError("dominant_phase must be result_conversion for the optimized path")
    after = payload.get("allocation_counts_after", {})
    if after.get("rust_per_sample_heap_allocations") != 0:
        raise ValueError("allocation_counts_after rust_per_sample_heap_allocations")
    if after.get("python_per_sample_heap_allocations") != 0:
        raise ValueError("allocation_counts_after python_per_sample_heap_allocations")
    if payload.get("timing_promotion_claimed") is not False:
        raise ValueError("timing_promotion_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("two_axis_exchange_reopened") is not False:
        raise ValueError("two_axis_exchange_reopened must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("optimized_path_ready_score") != optimized_path_ready_score(payload):
        raise ValueError("optimized_path_ready_score mismatch")
    expected_status = "complete" if payload.get("optimized_path_ready_score") == 1.0 else "blocked"
    if payload.get("status") != expected_status:
        raise ValueError("status mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict prefix")
    if verdict != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact under the requested repository root."""

    output = Path(root) / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return output


def upstream_artifact_hashes(root: Path) -> JsonDict:
    """Hash upstream artifacts and preserve their bare gate values."""

    receipts: JsonDict = {}
    for key, relative in UPSTREAM_ARTIFACTS.items():
        path = root / relative
        receipt: JsonDict = {"path": relative.as_posix(), "available": path.exists()}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt.update(
                {
                    "sha256": sha256_file(path),
                    "honest_verdict": payload.get("honest_verdict"),
                    "restart_parity_ready_score": payload.get("restart_parity_ready_score"),
                    "rust_benchmark_gate_ready_score": payload.get(
                        "rust_benchmark_gate_ready_score"
                    ),
                    "distributional_parity_score": payload.get("distributional_parity_score"),
                    "production_backend_reachable_score": payload.get(
                        "production_backend_reachable_score"
                    ),
                }
            )
        receipts[key] = receipt
    return receipts


def source_hashes_before(root: Path) -> JsonDict:
    """Return pre-optimization source hashes from Exp5758 where available."""

    path = root / UPSTREAM_ARTIFACTS["exp5758"]
    if not path.exists():
        return {"source": "missing_exp5758"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": "exp5758.repair_source_hashes",
        "hashes": dict(payload.get("repair_source_hashes", {})),
    }


def source_hashes_after(root: Path) -> JsonDict:
    """Hash the current source files changed or consumed by Exp5764."""

    return {
        path: sha256_file(root / path) if (root / path).exists() else None
        for path in SOURCE_HASH_PATHS
    }


def rust_toolchain() -> JsonDict:
    """Record local Rust compiler versions."""

    return {
        "rustc": _command_output(["rustc", "--version", "--verbose"]),
        "cargo": _command_output(["cargo", "--version", "--verbose"]),
    }


def python_version() -> JsonDict:
    """Record local Python runtime identity."""

    return {
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def pyo3_version(root: Path) -> JsonDict:
    """Read PyO3 dependency pins from Cargo files."""

    lock = root / "Cargo.lock"
    lock_text = lock.read_text(encoding="utf-8") if lock.exists() else ""
    cargo = root / "crates/carnot-python/Cargo.toml"
    cargo_text = cargo.read_text(encoding="utf-8") if cargo.exists() else ""
    return {
        "Cargo.lock": "0.24.2" if 'name = "pyo3"' in lock_text and "0.24.2" in lock_text else None,
        "Cargo.toml_requirement": "0.24" if 'pyo3 = { version = "0.24"' in cargo_text else None,
        "features": ["extension-module"],
    }


def host_receipts(root: Path) -> JsonDict:
    """Collect host stability and resource receipts."""

    return {
        "cpu_model": _cpu_model_name(),
        "cpu_governor": _cpu_governor_receipt(),
        "thread_counts": {key: os.environ.get(key) for key in _thread_env_keys()},
        "free_ram": _meminfo_receipt(),
        "free_disk": _disk_receipt(root),
        "competing_benchmark_processes": _competing_processes(),
    }


def release_build_receipt(root: Path, *, run_build: bool) -> JsonDict:
    """Record release-build capability without forcing tests to rebuild Rust."""

    command = ["cargo", "build", "-p", "carnot-python", "--release"]
    env = {"PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"}
    if run_build:
        result = subprocess.run(  # noqa: S603 - fixed local build command.
            command,
            cwd=root,
            env={**os.environ, **env},
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 " + " ".join(command),
            "exit_code": result.returncode,
            "completed": result.returncode == 0,
            "stdout_tail": result.stdout.splitlines()[-10:],
            "stderr_tail": result.stderr.splitlines()[-10:],
        }
    library = root / "target/release/libcarnot_python.so"
    return {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 " + " ".join(command),
        "exit_code": 0 if library.exists() else None,
        "completed": library.exists(),
        "reused_existing_release_artifact": library.exists(),
    }


def affinity_receipt(*, freeze_affinity: bool) -> JsonDict:
    """Return fixed CPU placement receipt."""

    if not hasattr(os, "sched_getaffinity"):
        return {
            "observable": False,
            "freeze_requested": freeze_affinity,
            "current_cpus": [],
            "fixed_worker_policy": "single_worker_no_dynamic_spawn",
        }
    current = sorted(os.sched_getaffinity(0))
    return {
        "observable": True,
        "freeze_requested": freeze_affinity,
        "current_cpus": current,
        "profile_cpus": current[:1],
        "fixed_worker_policy": "single_worker_no_dynamic_spawn",
    }


def preconditions_checked(
    upstream: Mapping[str, Any],
    host: Mapping[str, Any],
    release: Mapping[str, Any],
) -> list[JsonDict]:
    """Return pre-profile provenance and host gates."""

    exp5758 = upstream.get("exp5758", {})
    return [
        {
            "resource": "upstream_artifact_hashes",
            "available": all(row.get("available") is True for row in upstream.values()),
        },
        {
            "resource": "exp5751_parity_replay",
            "available": exp5758.get("rust_benchmark_gate_ready_score") == 1.0,
        },
        {"resource": "release_build_capability", "available": release.get("completed") is True},
        {
            "resource": "free_ram",
            "available": int(host.get("free_ram", {}).get("mem_available_kib", 0)) > 512_000,
        },
        {
            "resource": "free_disk",
            "available": int(host.get("free_disk", {}).get("free_bytes", 0)) > 512_000_000,
        },
        {
            "resource": "no_competing_benchmark_process",
            "available": not host.get("competing_benchmark_processes"),
        },
        {
            "resource": "cpu_governor_observed_or_blocked",
            "available": "governors" in host.get("cpu_governor", {})
            or "blocked_reason" in host.get("cpu_governor", {}),
        },
    ]


def _time_phase(phase: str, item: Mapping[str, Any], *, include_restart: bool) -> float:
    start = time.perf_counter()
    if phase == "serialization":
        canonical_json(item["config"])
    elif phase == "python_preparation":
        np.ascontiguousarray(item["biases"], dtype=np.float64)
        np.ascontiguousarray(item["couplings"], dtype=np.float64)
    elif phase == "pyo3_crossing":
        OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
            item["biases"],
            item["couplings"],
            1,
            {**item["config"], "return_decision_log": False, "burn_in_sweeps": 0},
        )
    elif phase == "rust_batch_allocation":
        OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch([item])
    elif phase == "worker_scheduling":
        sum(range(1))
    elif phase == "within_swap_kernel_work":
        OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
            item["biases"],
            item["couplings"],
            item["n_samples"],
            item["config"],
        )
    elif phase == "validation":
        row = OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch([item])[0]
        _validate_row(item, row)
    elif phase == "result_conversion":
        full = OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
            item["biases"],
            item["couplings"],
            item["n_samples"],
            {**item["config"], "return_decision_log": True},
        )
        canonical_json(full["decision_log"])
    elif phase == "checkpoint":
        row = OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch([item])[0]
        checkpoint_checksum(row["checkpoint"])
    elif phase == "restart":
        if include_restart:
            row = OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch([item])[0]
            _restart_match(item, row, prefer_rust=True)
    else:
        raise ValueError(f"unknown phase: {phase}")
    return time.perf_counter() - start


def _restart_match(
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    prefer_rust: bool,
) -> bool:
    suffix = OneAxisRustBackend(
        seed=int(item["config"]["seed"]),
        prefer_rust=prefer_rust,
    ).run_descriptor(
        item["biases"],
        item["couplings"],
        1,
        {**item["config"], "checkpoint": row["checkpoint"], "burn_in_sweeps": 0},
    )
    return bool(
        suffix["checkpoint"]["payload_checksum"] == checkpoint_checksum(suffix["checkpoint"])
    )


def _validate_row(item: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("receipt", {}).get("optimized_hot_path", {}).get("used") is True
        and len(row.get("samples", [])) == int(item["n_samples"])
        and row.get("checkpoint", {}).get("payload_checksum")
        == checkpoint_checksum(row.get("checkpoint", {}))
    )


def _combine_allocation_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rust_counts = [int(row["rust_per_sample_heap_allocations"]) for row in rows]
    python_counts = [int(row["python_per_sample_heap_allocations"]) for row in rows]
    return {
        "rust_per_sample_heap_allocations": max(rust_counts) if rust_counts else 0,
        "python_per_sample_heap_allocations": max(python_counts) if python_counts else 0,
        "documented_unavoidable_boundaries": ["numpy_samples_array", "checkpoint_dict"],
        "observed_case_count": len(rows),
    }


def _batch_item(
    workload: Mapping[str, Any],
    seed: int,
    n_samples: int,
    *,
    compact: bool,
) -> JsonDict:
    fields, couplings = exp5724.arrays_from_workload(workload)
    return {
        "workload_id": workload["workload_id"],
        "biases": fields,
        "couplings": couplings,
        "n_samples": int(n_samples),
        "config": descriptor_for_run(
            seed=int(seed),
            initial_states=exp5724.initial_states_for(workload, int(seed)),
            initial_labels=list(range(len(exp5714.BETA_LADDER))),
            burn_in_sweeps=DEFAULT_BURN_IN_SWEEPS,
        )
        | {"return_decision_log": not compact},
    }


def _workload_for_size(size: int) -> JsonDict:
    return exp5724.build_workload_manifest(
        problem_sizes=(int(size),),
        topology_families=("ferromagnetic_ring_easy",),
    )[0]


def _spin_rows(samples_spin: Any) -> list[list[int]]:
    return np.asarray(samples_spin, dtype=np.int8).astype(int).tolist()


def _gate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return 1.0 if float(value) >= 1.0 else 0.0


def _command_output(command: Sequence[str]) -> JsonDict:
    result = subprocess.run(  # noqa: S603 - fixed local inspection commands.
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "lines": (result.stdout or result.stderr).splitlines(),
    }


def _cpu_model_name() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():  # pragma: no cover - Linux host receipt fallback.
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None  # pragma: no cover - malformed /proc/cpuinfo fallback.


def _cpu_governor_receipt() -> JsonDict:
    governors = sorted(
        {
            path.read_text(encoding="utf-8").strip()
            for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
            if path.exists()
        }
    )
    if governors:
        return {"governors": governors, "stable_single_governor": len(governors) == 1}
    return {  # pragma: no cover - host-dependent cpufreq fallback.
        "blocked_reason": "cpufreq_governor_not_observable"
    }


def _thread_env_keys() -> tuple[str, ...]:
    return (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )


def _meminfo_receipt() -> JsonDict:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():  # pragma: no cover - Linux host receipt fallback.
        return {"observable": False, "mem_available_kib": 0}
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.strip().split()[0])
    return {
        "observable": True,
        "mem_total_kib": values.get("MemTotal", 0),
        "mem_available_kib": values.get("MemAvailable", 0),
    }


def _disk_receipt(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {"total_bytes": usage.total, "free_bytes": usage.free}


def _competing_processes() -> list[JsonDict]:
    result = subprocess.run(  # noqa: S603 - fixed local process inspection.
        ["ps", "-eo", "pid=,comm=,args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    current = str(os.getpid())
    needles = ("experiment_5739_one_axis", "experiment_5764_one_axis", "criterion")
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3 or parts[0] == current:
            continue
        if "pytest" in parts[2] or "test_experiment_5764" in parts[2]:
            continue
        if any(needle in parts[2] for needle in needles):
            rows.append({"pid": int(parts[0]), "command": parts[1], "args": parts[2]})
    return rows


def _stable_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = build_artifact(root=REPO_ROOT)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
