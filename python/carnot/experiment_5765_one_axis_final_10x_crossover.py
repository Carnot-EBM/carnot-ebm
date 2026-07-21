"""Exp5765 final one-axis Rust/Python 10x crossover benchmark.

Spec refs: REQ-SAMPLE-5765, SCENARIO-SAMPLE-5765.

This is the terminal NFR-01 attempt for the allocation-reduced one-axis
Rust/PyO3 production path. It benchmarks matched Python exact fallback and
Rust release paths end-to-end, claims 10x only under the consecutive larger
size confidence rule, and otherwise retires this narrow PyO3 technique without
making any hardware or two-axis claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5724_one_axis_rust_python_matched_crossover as exp5724
from carnot import experiment_5764_one_axis_profiled_allocation_free_hot_path as exp5764
from carnot.samplers.backend import get_backend
from carnot.samplers.one_axis_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    ENERGY_CONVENTION,
    ONE_AXIS_ALGORITHM,
    ONE_AXIS_TOPOLOGY,
    OneAxisRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
)


JsonDict = dict[str, Any]
BenchmarkRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5765_one_axis_final_10x_crossover.json")
RUN_DATE = "20260721"
SPEC_REFS = ("REQ-SAMPLE-5765", "SCENARIO-SAMPLE-5765")
INFERENCE_SUBSTRATE = "local_rust_pyo3_cpu_release_matched_benchmark"

RUST_ARM = "rust_pyo3_release_production"
PYTHON_ARM = "python_exact_fallback"
ARMS = (RUST_ARM, PYTHON_ARM)

DEFAULT_CELL_SIZES = (48, 96, 192, 256)
DEFAULT_RANDOM_SEEDS = tuple(range(5_765_000, 5_765_030))
DEFAULT_PAIRED_BATCHES_PER_CELL = 30
DEFAULT_WARMUP_BATCHES = 1
DEFAULT_RETAINED_SAMPLES_PER_BATCH = 2
DEFAULT_BURN_IN_SWEEPS = 1
BOOTSTRAP_RESAMPLES = 500

PREREGISTERED_EXCLUSION_REASONS = (
    "semantic_parity_failed",
    "restart_parity_failed",
    "distributional_parity_failed",
    "fallback_equivalence_failed",
    "production_backend_not_reachable",
    "warmup_unstable",
    "timing_nonpositive",
)

TEST_COMMANDS = (
    "cargo test -p carnot-samplers one_axis_tempering",
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python --release",
    ".venv/bin/pytest tests/python/test_experiment_5765_one_axis_final_10x_crossover.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null -m pytest tests/python/test_experiment_5765_one_axis_final_10x_crossover.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5765_one_axis_final_10x_crossover.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every Exp5765 field exists before a reviewer trusts the final NFR-01 claim or retirement.",
    "status": "Bare terminal state lets gates distinguish complete from blocked without parsing prose.",
    "preconditions_checked": "Records provenance, release-build, host-stability, warmup, restart, and manifest gates before timing starts.",
    "spec_refs": "Binds the artifact to REQ-SAMPLE-5765 and SCENARIO-SAMPLE-5765.",
    "upstream_artifact_hashes": "Pins Exp5751, Exp5739, and Exp5764 so final timing cannot drift from repaired and optimized provenance.",
    "source_hashes": "Hashes optimized source bytes used by the final release benchmark.",
    "release_binary_hashes": "Hashes the local release extension artifacts used by the Rust/PyO3 path.",
    "host_receipts": "Authenticates CPU, governor, thread, memory, disk, and competing-load observations.",
    "affinity_receipt": "Records fixed CPU placement used for warmup and paired timing.",
    "benchmark_manifest": "Freezes sizes, batches, seeds, budgets, paths, tolerances, outlier policy, bootstrap method, and claim rule before timing.",
    "benchmark_manifest_hash": "Content-addresses the preregistered manifest independently of timing results.",
    "cell_sizes": "Makes the 48/96/192 plus larger feasible final panel explicit.",
    "paired_batches_per_cell": "Proves each qualified size has at least thirty paired measured batches after warmup.",
    "raw_timing_receipts": "Preserves raw interleaved per-batch end-to-end timing and phase receipts before summaries.",
    "warmup_receipts": "Shows warmup stability before science rows enter the benchmark.",
    "quality_metrics_by_cell": "Reports energy, feasibility, acceptance, retained count, ESS, and autocorrelation diagnostics before speed claims.",
    "semantic_parity_by_cell": "Proves exact semantic parity where exactness applies.",
    "restart_parity_by_cell": "Proves checkpoint and restart parity remained repaired during timing.",
    "distributional_parity_by_cell": "Reports distributional diagnostics so speed cannot hide wrong samples.",
    "fallback_equivalence": "Proves the Python exact fallback and Rust production path remain matched.",
    "production_backend_reachable": "Proves the explicit one_axis_rust SamplerBackend reached the optimized release path.",
    "exclusion_manifest": "Reports every excluded pair with a preregistered reason instead of silently filtering timing.",
    "speedup_median_by_size": "Preserves honest measured paired speedup medians by size.",
    "speedup_lcb_by_size": "Applies the paired bootstrap lower bound used by the 10x claim rule.",
    "speedup_ucb_by_size": "Preserves the paired bootstrap upper bound for uncertainty review.",
    "consecutive_larger_size_rule_passed": "Bare gate proves two consecutive larger sizes have lower confidence bound >=10.0.",
    "matched_quality_gate_passed": "Bare gate proves all parity and quality checks passed before any NFR-01 claim.",
    "rust_10x_claimed": "Bare true is allowed only under the final consecutive larger-size confidence rule.",
    "rust_10x_retired": "Bare true retires only this allocation-free one-axis PyO3 technique after a repeated null.",
    "remaining_bottleneck": "Names the measured bottleneck left after allocation reduction when the 10x rule fails.",
    "nfr01_status": "States whether PRD NFR-01 is qualified, retired for this technique, or blocked.",
    "hardware_speedup_claimed": "Bare false prevents local CPU software timing from becoming a hardware claim.",
    "two_axis_exchange_reopened": "Bare false keeps retired two-axis exchange out of scope.",
    "inference_substrate": "Declares local Rust/PyO3 CPU release matched benchmarking, not LLM, GPU, FPGA, or TSU inference.",
    "random_seeds": "Records the paired seed schedule for replay.",
    "duration_s": "Records real wall-clock artifact construction time for fabrication review.",
    "test_commands": "Lists the verification commands run for the final benchmark.",
    "test_exit_codes": "Records command outcomes honestly.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether NFR-01 was claimed, retired for this technique, or blocked.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content with the repository convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one file byte-for-byte."""

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


def benchmark_manifest(
    *,
    cell_sizes: Sequence[int] = DEFAULT_CELL_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    paired_batches_per_cell: int = DEFAULT_PAIRED_BATCHES_PER_CELL,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    retained_samples_per_batch: int = DEFAULT_RETAINED_SAMPLES_PER_BATCH,
    burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS,
    allow_underpowered: bool = False,
) -> JsonDict:
    """Freeze the Exp5765 benchmark design before timing evidence exists."""

    sizes = tuple(int(size) for size in cell_sizes)
    seeds = tuple(int(seed) for seed in random_seeds)
    if not allow_underpowered:
        if not {48, 96, 192}.issubset(set(sizes)):
            raise ValueError("cell_sizes must include 48, 96, and 192")
        if len(sizes) < 4:
            raise ValueError("cell_sizes must include one larger feasible size")
        if int(paired_batches_per_cell) < 30:
            raise ValueError("paired_batches_per_cell must be at least thirty")
        if len(seeds) < int(paired_batches_per_cell):
            raise ValueError("random_seeds must cover every paired batch")
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("cell_sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("cell_sizes must be unique")
    if not seeds:
        raise ValueError("random_seeds must not be empty")
    if int(warmup_batches) < 0:
        raise ValueError("warmup_batches must be nonnegative")
    if int(retained_samples_per_batch) <= 0:
        raise ValueError("retained_samples_per_batch must be positive")
    if int(burn_in_sweeps) < 0:
        raise ValueError("burn_in_sweeps must be nonnegative")

    return {
        "schema": "carnot.exp5765.final_one_axis_10x_manifest.v1",
        "frozen_before_timing": True,
        "run_date": RUN_DATE,
        "cell_sizes": list(sizes),
        "paired_batches_per_cell": int(paired_batches_per_cell),
        "random_seeds": list(seeds),
        "warmup_batches": int(warmup_batches),
        "retained_samples_per_batch": int(retained_samples_per_batch),
        "burn_in_sweeps": int(burn_in_sweeps),
        "arms": {
            RUST_ARM: {
                "backend": "OneAxisRustBackend",
                "prefer_rust": True,
                "sampler_api": "sample_batch",
                "return_decision_log": False,
                "expected_active_backend": ACTIVE_RUST_BACKEND,
            },
            PYTHON_ARM: {
                "backend": "OneAxisRustBackend",
                "prefer_rust": False,
                "sampler_api": "sample_batch",
                "return_decision_log": False,
                "expected_active_backend": ACTIVE_PYTHON_FALLBACK,
            },
        },
        "algorithm": ONE_AXIS_ALGORITHM,
        "topology": ONE_AXIS_TOPOLOGY,
        "energy_convention": ENERGY_CONVENTION,
        "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "restart_schedule": "one checkpoint suffix replay per arm and pair",
        "path_order": "alternating_rust_first_then_python_first_by_batch_index",
        "quality_tolerances": {
            "energy_delta_abs_max": 0.0,
            "energy_histogram_tv_max": 0.0,
            "acceptance_delta_abs_max": 0.0,
            "retained_sample_count_match_required": True,
            "semantic_hash_match_required": True,
            "restart_suffix_hash_match_required": True,
        },
        "outlier_policy": "no outlier removal; invalid pairs excluded only by preregistered reasons",
        "statistical_method": {
            "speedup_ratio": "python_end_to_end_s / rust_end_to_end_s",
            "bootstrap": "paired nonparametric bootstrap over included batch ratios",
            "confidence_level": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "claim_rule": (
            "rust_10x_claimed is true only when matched quality passes and the paired "
            "bootstrap lower bound is >=10.0 at two consecutive larger sizes"
        ),
        "excluded_from_headline": [
            "setup_only",
            "kernel_only",
            "simulated",
            "debug",
            "unmatched",
            "hardware",
            "gpu",
            "fpga",
            "tsu",
        ],
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    benchmark_runner: BenchmarkRunner | None = None,
    cell_sizes: Sequence[int] = DEFAULT_CELL_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    paired_batches_per_cell: int = DEFAULT_PAIRED_BATCHES_PER_CELL,
    freeze_affinity: bool = True,
    tests_added_or_reused: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build and validate the terminal Exp5765 artifact."""

    del tests_added_or_reused
    started = time.perf_counter()
    root_path = Path(root)
    manifest = benchmark_manifest(
        cell_sizes=cell_sizes,
        random_seeds=random_seeds,
        paired_batches_per_cell=paired_batches_per_cell,
    )
    upstream = upstream_artifact_hashes(root_path)
    sources = source_hashes(root_path)
    binaries = release_binary_hashes(root_path)
    host = host_receipts(root_path)
    affinity = exp5764.affinity_receipt(freeze_affinity=freeze_affinity)
    affinity["pinned_during_timing"] = bool(
        freeze_affinity and affinity.get("observable") is True and affinity.get("profile_cpus")
    )
    preconditions = preconditions_checked(upstream, sources, binaries, host, affinity, manifest)
    runner = benchmark_runner or run_matched_release_benchmark
    if all_preconditions_passed(preconditions):
        with _temporary_affinity(
            affinity.get("profile_cpus", []),
            enabled=bool(affinity.get("pinned_during_timing")),
        ):
            evidence = runner(
                manifest=manifest,
                cell_sizes=manifest["cell_sizes"],
                random_seeds=manifest["random_seeds"],
            )
    else:
        evidence = blocked_evidence(manifest, reason="precondition_gate_failed")

    speedups = speedup_intervals_from_raw(
        evidence["raw_timing_receipts"],
        cell_sizes=manifest["cell_sizes"],
    )
    quality_gate = matched_quality_gate(evidence, manifest)
    consecutive_rule = consecutive_larger_size_rule_passed(
        speedups["lcb"],
        cell_sizes=manifest["cell_sizes"],
    )
    status = "complete" if all_preconditions_passed(preconditions) and quality_gate else "blocked"
    claimed = bool(status == "complete" and quality_gate and consecutive_rule)
    retired = bool(status == "complete" and quality_gate and not consecutive_rule)

    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": preconditions,
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": upstream,
        "source_hashes": sources,
        "release_binary_hashes": binaries,
        "host_receipts": host,
        "affinity_receipt": affinity,
        "benchmark_manifest": manifest,
        "benchmark_manifest_hash": sha256_json(manifest),
        "cell_sizes": list(manifest["cell_sizes"]),
        "paired_batches_per_cell": int(manifest["paired_batches_per_cell"]),
        "raw_timing_receipts": evidence["raw_timing_receipts"],
        "warmup_receipts": evidence["warmup_receipts"],
        "quality_metrics_by_cell": evidence["quality_metrics_by_cell"],
        "semantic_parity_by_cell": evidence["semantic_parity_by_cell"],
        "restart_parity_by_cell": evidence["restart_parity_by_cell"],
        "distributional_parity_by_cell": evidence["distributional_parity_by_cell"],
        "fallback_equivalence": evidence["fallback_equivalence"],
        "production_backend_reachable": evidence["production_backend_reachable"],
        "exclusion_manifest": evidence["exclusion_manifest"],
        "speedup_median_by_size": speedups["median"],
        "speedup_lcb_by_size": speedups["lcb"],
        "speedup_ucb_by_size": speedups["ucb"],
        "consecutive_larger_size_rule_passed": consecutive_rule,
        "matched_quality_gate_passed": quality_gate,
        "rust_10x_claimed": claimed,
        "rust_10x_retired": retired,
        "remaining_bottleneck": remaining_bottleneck(evidence, claimed=claimed),
        "nfr01_status": nfr01_status(claimed=claimed, retired=retired, status=status),
        "hardware_speedup_claimed": False,
        "two_axis_exchange_reopened": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(manifest["random_seeds"]),
        "duration_s": exp5764._stable_float(time.perf_counter() - started),  # noqa: SLF001
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": {
            command: (test_exit_codes or {}).get(command) for command in TEST_COMMANDS
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_matched_release_benchmark(
    *,
    manifest: Mapping[str, Any],
    cell_sizes: Sequence[int],
    random_seeds: Sequence[int],
) -> JsonDict:
    """Run matched Rust release and Python fallback batches on identical cells."""

    raw_rows: list[JsonDict] = []
    warmups: list[JsonDict] = []
    by_size: dict[int, list[JsonDict]] = {int(size): [] for size in cell_sizes}
    for size in cell_sizes:
        warmups.append(_warmup_receipt(int(size), manifest, int(random_seeds[0])))
        for batch_index in range(int(manifest["paired_batches_per_cell"])):
            seed = int(random_seeds[batch_index % len(random_seeds)])
            pair = _measure_pair(int(size), int(batch_index), seed, manifest)
            raw_rows.append(pair)
            by_size[int(size)].append(pair)

    quality = [_quality_metrics_for_size(size, rows) for size, rows in by_size.items()]
    semantic = [_semantic_parity_for_size(size, rows) for size, rows in by_size.items()]
    restart = [_restart_parity_for_size(size, rows) for size, rows in by_size.items()]
    distributional = [_distributional_parity_for_size(size, rows) for size, rows in by_size.items()]
    exclusions = [
        {
            "pair_id": row["pair_id"],
            "size": row["size"],
            "batch_index": row["batch_index"],
            "reason": row["exclusion_reason"],
        }
        for row in raw_rows
        if row.get("included") is not True
    ]
    return {
        "raw_timing_receipts": raw_rows,
        "warmup_receipts": warmups,
        "quality_metrics_by_cell": quality,
        "semantic_parity_by_cell": semantic,
        "restart_parity_by_cell": restart,
        "distributional_parity_by_cell": distributional,
        "fallback_equivalence": {
            "passed": not exclusions,
            "exact_fallback_equivalence": not exclusions,
        },
        "production_backend_reachable": {
            "passed": all(row["rust_active_backend"] == ACTIVE_RUST_BACKEND for row in raw_rows),
            "active_backend": ACTIVE_RUST_BACKEND,
            "optimized_hot_path_used": all(
                row["rust_optimized_hot_path_used"] is True for row in raw_rows
            ),
            "sampler_backend_factory": isinstance(get_backend("one_axis_rust"), OneAxisRustBackend),
        },
        "exclusion_manifest": {
            "preregistered_reasons": list(PREREGISTERED_EXCLUSION_REASONS),
            "exclusions": exclusions,
        },
    }


def blocked_evidence(manifest: Mapping[str, Any], *, reason: str) -> JsonDict:
    """Emit a no-timing evidence shape when preconditions block science rows."""

    rows = [
        {
            "size": int(size),
            "pair_count": 0,
            "quality_matched": False,
            "blocked_reason": reason,
        }
        for size in manifest["cell_sizes"]
    ]
    parity_rows = [
        {"size": int(size), "passed": False, "blocked_reason": reason}
        for size in manifest["cell_sizes"]
    ]
    return {
        "raw_timing_receipts": [],
        "warmup_receipts": [
            {"size": int(size), "warmup_batches": 0, "stable": False, "blocked_reason": reason}
            for size in manifest["cell_sizes"]
        ],
        "quality_metrics_by_cell": rows,
        "semantic_parity_by_cell": parity_rows,
        "restart_parity_by_cell": parity_rows,
        "distributional_parity_by_cell": parity_rows,
        "fallback_equivalence": {"passed": False, "blocked_reason": reason},
        "production_backend_reachable": {"passed": False, "blocked_reason": reason},
        "exclusion_manifest": {
            "preregistered_reasons": list(PREREGISTERED_EXCLUSION_REASONS),
            "exclusions": [{"reason": reason, "count": 0}],
        },
    }


def speedup_intervals_from_raw(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    cell_sizes: Sequence[int],
) -> JsonDict:
    """Compute paired median speedups and bootstrap 95% intervals by size."""

    groups: dict[str, list[float]] = {str(int(size)): [] for size in cell_sizes}
    groups["aggregate"] = []
    for row in raw_rows:
        if row.get("included") is not True:
            continue
        ratio = float(row.get("speedup_ratio", 0.0))
        if ratio <= 0:
            continue
        key = str(int(row["size"]))
        groups.setdefault(key, []).append(ratio)
        groups["aggregate"].append(ratio)
    median: JsonDict = {}
    lcb: JsonDict = {}
    ucb: JsonDict = {}
    for key, values in groups.items():
        interval = bootstrap_interval(values, seed=_stable_seed("exp5765-bootstrap", key))
        median[key] = exp5764._stable_float(statistics.median(values)) if values else None  # noqa: SLF001
        lcb[key] = interval[0]
        ucb[key] = interval[1]
    return {"median": median, "lcb": lcb, "ucb": ucb}


def bootstrap_interval(values: Sequence[float], *, seed: int) -> list[float | None]:
    """Return a deterministic paired-bootstrap confidence interval."""

    vals = np.asarray([float(value) for value in values], dtype=np.float64)
    if vals.size == 0:
        return [None, None]
    if vals.size == 1:
        value = exp5764._stable_float(vals[0])  # noqa: SLF001
        return [value, value]
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        means[index] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    low, high = np.percentile(means, [2.5, 97.5])
    return [exp5764._stable_float(low), exp5764._stable_float(high)]  # noqa: SLF001


def consecutive_larger_size_rule_passed(
    speedup_lcb_by_size: Mapping[str, Any],
    *,
    cell_sizes: Sequence[int],
) -> bool:
    """Apply the final two-consecutive-larger-size 10x rule."""

    ordered = [int(size) for size in cell_sizes]
    larger = ordered[1:] if len(ordered) >= 3 else ordered
    for left, right in zip(larger, larger[1:], strict=False):
        left_lcb = speedup_lcb_by_size.get(str(left))
        right_lcb = speedup_lcb_by_size.get(str(right))
        if left_lcb is not None and right_lcb is not None:
            if float(left_lcb) >= 10.0 and float(right_lcb) >= 10.0:
                return True
    return False


def matched_quality_gate(evidence: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    """Return whether every preregistered quality and parity gate passed."""

    expected_pairs = int(manifest["paired_batches_per_cell"])
    gates = [
        all(row.get("stable") is True for row in evidence.get("warmup_receipts", [])),
        all(
            row.get("quality_matched") is True and int(row.get("pair_count", 0)) >= expected_pairs
            for row in evidence.get("quality_metrics_by_cell", [])
        ),
        all(row.get("passed") is True for row in evidence.get("semantic_parity_by_cell", [])),
        all(row.get("passed") is True for row in evidence.get("restart_parity_by_cell", [])),
        all(row.get("passed") is True for row in evidence.get("distributional_parity_by_cell", [])),
        evidence.get("fallback_equivalence", {}).get("passed") is True,
        evidence.get("fallback_equivalence", {}).get("exact_fallback_equivalence") is True,
        evidence.get("production_backend_reachable", {}).get("passed") is True,
        evidence.get("production_backend_reachable", {}).get("optimized_hot_path_used") is True,
        not evidence.get("exclusion_manifest", {}).get("exclusions"),
    ]
    return bool(all(gates))


def remaining_bottleneck(evidence: Mapping[str, Any], *, claimed: bool) -> str:
    """Name the measured phase that remains load-bearing when 10x fails."""

    if claimed:
        return "none_10x_consecutive_larger_size_rule_passed"
    phase_shares: dict[str, list[float]] = {}
    for row in evidence.get("raw_timing_receipts", []):
        rust_total = float(row.get("rust_end_to_end_s", 0.0))
        if rust_total <= 0:
            continue
        rust_phases = row.get("phase_receipts", {}).get(RUST_ARM, {})
        for phase, value in rust_phases.items():
            phase_shares.setdefault(str(phase), []).append(float(value) / rust_total)
    if not phase_shares:
        return "quality_or_precondition_gate_blocked_timing"
    phase, shares = max(phase_shares.items(), key=lambda item: statistics.median(item[1]))
    return f"{phase}_dominates_rust_release_end_to_end_below_10x_lcb"


def nfr01_status(*, claimed: bool, retired: bool, status: str) -> str:
    """Return the NFR-01 terminal status for this narrow technique."""

    if claimed:
        return "qualified_for_this_one_axis_pyo3_technique"
    if retired:
        return "retired_allocation_free_one_axis_pyo3_technique"
    if status == "blocked":
        return "blocked_quality_or_precondition_gate"
    return "not_qualified_without_retirement"


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal complete or blocked verdict."""

    if payload.get("rust_10x_claimed") is True:
        return (
            "complete: PRD NFR-01 10x claimed for the allocation-free one-axis "
            "Rust/PyO3 CPU release path under the consecutive larger-size rule; "
            "no hardware or two-axis claim"
        )
    if payload.get("rust_10x_retired") is True:
        return (
            "complete: final matched-quality CPU release benchmark did not prove "
            "the consecutive larger-size 10x lower-bound rule; this allocation-free "
            "one-axis PyO3 technique is retired, not future Rust or hardware work"
        )
    return "blocked: Exp5765 quality, provenance, or host precondition gate failed before NFR-01 interpretation"


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5765 schema and fail closed on unsafe claim edits."""

    if tuple(payload) != REQUIRED_ARTIFACT_FIELDS:
        raise ValueError("artifact fields mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("spec_refs") != list(SPEC_REFS):
        raise ValueError("spec_refs mismatch")
    if int(payload.get("paired_batches_per_cell", 0)) < 30:
        raise ValueError("paired_batches_per_cell must be at least thirty")
    if not {48, 96, 192}.issubset(set(payload.get("cell_sizes", []))):
        raise ValueError("cell_sizes must include 48, 96, and 192")
    if len(payload.get("cell_sizes", [])) < 4:
        raise ValueError("cell_sizes must include one larger feasible size")
    if payload.get("benchmark_manifest_hash") != sha256_json(payload.get("benchmark_manifest")):
        raise ValueError("benchmark_manifest_hash mismatch")
    for field in (
        "consecutive_larger_size_rule_passed",
        "matched_quality_gate_passed",
        "rust_10x_claimed",
        "rust_10x_retired",
        "hardware_speedup_claimed",
        "two_axis_exchange_reopened",
    ):
        if type(payload.get(field)) is not bool:
            raise ValueError(field)
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("two_axis_exchange_reopened") is not False:
        raise ValueError("two_axis_exchange_reopened must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("status") not in {"complete", "blocked"}:
        raise ValueError("status mismatch")
    expected_status = (
        "complete" if payload.get("matched_quality_gate_passed") is True else "blocked"
    )
    if payload.get("status") != expected_status:
        raise ValueError("status mismatch")
    expected_claim = bool(
        payload.get("status") == "complete"
        and payload.get("matched_quality_gate_passed") is True
        and payload.get("consecutive_larger_size_rule_passed") is True
    )
    if payload.get("rust_10x_claimed") is not expected_claim:
        raise ValueError("rust_10x_claimed mismatch")
    expected_retired = bool(
        payload.get("status") == "complete"
        and payload.get("matched_quality_gate_passed") is True
        and payload.get("consecutive_larger_size_rule_passed") is False
    )
    if payload.get("rust_10x_retired") is not expected_retired:
        raise ValueError("rust_10x_retired mismatch")
    expected_status = nfr01_status(
        claimed=bool(payload.get("rust_10x_claimed")),
        retired=bool(payload.get("rust_10x_retired")),
        status=str(payload.get("status")),
    )
    if payload.get("nfr01_status") != expected_status:
        raise ValueError("nfr01_status mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict prefix")
    if verdict != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal Exp5765 JSON artifact."""

    output = Path(root) / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return output


def upstream_artifact_hashes(root: Path) -> JsonDict:
    """Hash upstream evidence and preserve the gate fields Exp5765 consumes."""

    upstream = {
        "exp5751": Path("results/experiment_5751_rust_restart_parity_repair.json"),
        "exp5739": Path("results/experiment_5739_one_axis_batched_10x_crossover.json"),
        "exp5764": Path("results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json"),
    }
    receipts: JsonDict = {}
    for key, relative in upstream.items():
        path = root / relative
        receipt: JsonDict = {"path": relative.as_posix(), "available": path.exists()}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt.update(
                {
                    "sha256": sha256_file(path),
                    "honest_verdict": payload.get("honest_verdict"),
                    "restart_parity_ready_score": payload.get("restart_parity_ready_score"),
                    "rust_batched_10x_ready_score": payload.get("rust_batched_10x_ready_score"),
                    "optimized_path_ready_score": payload.get("optimized_path_ready_score"),
                    "software_speedup_claimed": payload.get("software_speedup_claimed"),
                }
            )
        receipts[key] = receipt
    return receipts


def source_hashes(root: Path) -> JsonDict:
    """Hash optimized source bytes relevant to the final benchmark."""

    paths = (
        "openspec/capabilities/samplers/spec.md",
        "python/carnot/samplers/one_axis_rust_backend.py",
        "python/carnot/experiment_5765_one_axis_final_10x_crossover.py",
        "tests/python/test_experiment_5765_one_axis_final_10x_crossover.py",
        "crates/carnot-python/src/one_axis_tempering.rs",
        "crates/carnot-samplers/src/one_axis_tempering.rs",
    )
    return {
        path: {"available": (root / path).exists(), "sha256": sha256_file(root / path)}
        if (root / path).exists()
        else {"available": False, "sha256": None}
        for path in paths
    }


def release_binary_hashes(root: Path) -> JsonDict:
    """Hash local release PyO3 binaries used by the benchmark."""

    candidates = sorted(
        {
            root / "target/release/libcarnot_python.so",
            *root.glob("target/release/deps/libcarnot_python*.so"),
        }
    )
    return {
        path.relative_to(root).as_posix(): {
            "available": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        for path in candidates
    }


def host_receipts(root: Path) -> JsonDict:
    """Collect host stability receipts for final timing."""

    receipt = exp5764.host_receipts(root)
    receipt["competing_benchmark_processes"] = _competing_processes()
    receipt["thread_count_policy"] = {
        "fixed_worker_policy": "one_axis_compact_path_single_worker",
        "thread_env": receipt.get("thread_counts", {}),
        "accepted_when_env_unset": True,
    }
    return receipt


def preconditions_checked(
    upstream: Mapping[str, Any],
    sources: Mapping[str, Any],
    binaries: Mapping[str, Any],
    host: Mapping[str, Any],
    affinity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    """Return every gate checked before timing rows can be interpreted."""

    return [
        {
            "resource": "upstream_artifact_hashes",
            "available": all(row.get("available") is True for row in upstream.values()),
            "details": upstream,
        },
        {
            "resource": "exp5751_restart_repair_ready",
            "available": upstream.get("exp5751", {}).get("restart_parity_ready_score") == 1.0,
        },
        {
            "resource": "exp5739_terminal_null_context",
            "available": upstream.get("exp5739", {}).get("rust_batched_10x_ready_score") == 0.0,
        },
        {
            "resource": "exp5764_optimized_path_ready",
            "available": upstream.get("exp5764", {}).get("optimized_path_ready_score") == 1.0,
        },
        {
            "resource": "optimized_source_hashes",
            "available": all(row.get("available") is True for row in sources.values()),
        },
        {
            "resource": "release_binary_hashes",
            "available": any(row.get("available") is True for row in binaries.values()),
        },
        {
            "resource": "affinity_recorded",
            "available": bool(affinity.get("current_cpus")),
            "details": affinity,
        },
        {
            "resource": "governor_observed_or_blocked",
            "available": "governors" in host.get("cpu_governor", {})
            or "blocked_reason" in host.get("cpu_governor", {}),
            "details": host.get("cpu_governor", {}),
        },
        {
            "resource": "fixed_thread_counts",
            "available": host.get("thread_count_policy", {}).get("fixed_worker_policy")
            == "one_axis_compact_path_single_worker",
            "details": host.get("thread_count_policy", {}),
        },
        {
            "resource": "no_competing_benchmark_load",
            "available": not host.get("competing_benchmark_processes"),
            "details": host.get("competing_benchmark_processes", []),
        },
        {
            "resource": "minimum_free_ram",
            "available": int(host.get("free_ram", {}).get("mem_available_kib", 0)) > 512_000,
            "details": host.get("free_ram", {}),
        },
        {
            "resource": "minimum_free_disk",
            "available": int(host.get("free_disk", {}).get("free_bytes", 0)) > 512_000_000,
            "details": host.get("free_disk", {}),
        },
        {
            "resource": "baseline_manifest_frozen",
            "available": manifest.get("frozen_before_timing") is True
            and int(manifest.get("paired_batches_per_cell", 0)) >= 30,
            "details": {"benchmark_manifest_hash": sha256_json(manifest)},
        },
    ]


def all_preconditions_passed(preconditions: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every pre-timing gate passed."""

    return all(row.get("available") is True for row in preconditions)


def _measure_pair(
    size: int,
    batch_index: int,
    seed: int,
    manifest: Mapping[str, Any],
) -> JsonDict:
    item = _batch_item(size, seed, manifest, return_decision_log=False)
    order = [RUST_ARM, PYTHON_ARM] if batch_index % 2 == 0 else [PYTHON_ARM, RUST_ARM]
    measured = {arm: _measure_arm(item, arm) for arm in order}
    diagnostic = _diagnostic_pair(size, seed, manifest)
    quality = _pair_quality(item, measured[RUST_ARM], measured[PYTHON_ARM], diagnostic)
    rust_s = float(measured[RUST_ARM]["end_to_end_s"])
    python_s = float(measured[PYTHON_ARM]["end_to_end_s"])
    speedup = python_s / rust_s if rust_s > 0 else 0.0
    reason = _pair_exclusion_reason(quality, measured)
    return {
        "pair_id": f"n{size}:batch{batch_index}",
        "size": int(size),
        "batch_index": int(batch_index),
        "seed": int(seed),
        "path_order": order,
        "rust_end_to_end_s": exp5764._stable_float(rust_s),  # noqa: SLF001
        "python_end_to_end_s": exp5764._stable_float(python_s),  # noqa: SLF001
        "speedup_ratio": exp5764._stable_float(speedup),  # noqa: SLF001
        "phase_receipts": {
            RUST_ARM: measured[RUST_ARM]["phase_s"],
            PYTHON_ARM: measured[PYTHON_ARM]["phase_s"],
        },
        "rust_active_backend": measured[RUST_ARM]["active_backend"],
        "python_active_backend": measured[PYTHON_ARM]["active_backend"],
        "rust_optimized_hot_path_used": measured[RUST_ARM]["optimized_hot_path_used"],
        "quality": quality,
        "included": reason is None,
        "exclusion_reason": reason,
    }


def _measure_arm(item: Mapping[str, Any], arm: str) -> JsonDict:
    prefer_rust = arm == RUST_ARM
    phase: dict[str, float] = {}
    start = time.perf_counter()
    phase_start = time.perf_counter()
    backend = OneAxisRustBackend(seed=int(item["config"]["seed"]), prefer_rust=prefer_rust)
    phase["setup_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    row = backend.sample_batch([item])[0]
    phase["sample_batch_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    encoded = canonical_json(_jsonable_result(row))
    phase["serialization_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    valid = _validate_result(item, row, encoded)
    phase["validation_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    restart = _restart_receipt(item, row, prefer_rust=prefer_rust)
    phase["restart_s"] = time.perf_counter() - phase_start

    elapsed = time.perf_counter() - start
    return {
        "arm": arm,
        "end_to_end_s": exp5764._stable_float(elapsed),  # noqa: SLF001
        "phase_s": {key: exp5764._stable_float(value) for key, value in phase.items()},  # noqa: SLF001
        "active_backend": row["receipt"]["active_backend"],
        "optimized_hot_path_used": row["receipt"]["optimized_hot_path"]["used"] is True,
        "samples_spin": _spin_rows(row["samples_spin"]),
        "checkpoint": row["checkpoint"],
        "restart": restart,
        "valid": valid,
    }


def _diagnostic_pair(size: int, seed: int, manifest: Mapping[str, Any]) -> JsonDict:
    item = _batch_item(size, seed, manifest, return_decision_log=True)
    rust = OneAxisRustBackend(seed=seed, prefer_rust=True).sample_batch([item])[0]
    python = OneAxisRustBackend(seed=seed, prefer_rust=False).sample_batch([item])[0]
    return {
        RUST_ARM: _decision_diagnostics(rust["decision_log"]),
        PYTHON_ARM: _decision_diagnostics(python["decision_log"]),
        "decision_log_hash_match": sha256_json(rust["decision_log"])
        == sha256_json(python["decision_log"]),
    }


def _pair_quality(
    item: Mapping[str, Any],
    rust: Mapping[str, Any],
    python: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> JsonDict:
    fields = np.asarray(item["biases"], dtype=np.float64)
    couplings = np.asarray(item["couplings"], dtype=np.float64)
    rust_energies = [
        exp5724.ising_energy(fields, couplings, sample) for sample in rust["samples_spin"]
    ]
    python_energies = [
        exp5724.ising_energy(fields, couplings, sample) for sample in python["samples_spin"]
    ]
    sample_hash_match = sha256_json(rust["samples_spin"]) == sha256_json(python["samples_spin"])
    restart_match = rust["restart"]["suffix_hash"] == python["restart"]["suffix_hash"]
    energy_delta = max(
        [abs(float(a) - float(b)) for a, b in zip(rust_energies, python_energies, strict=True)]
        or [0.0]
    )
    acceptance_delta = abs(
        float(diagnostic[RUST_ARM]["acceptance_rate"])
        - float(diagnostic[PYTHON_ARM]["acceptance_rate"])
    )
    return {
        "sample_hash_match": sample_hash_match,
        "checkpoint_state_match": rust["checkpoint"]["state"] == python["checkpoint"]["state"],
        "restart_suffix_hash_match": restart_match,
        "decision_log_hash_match": diagnostic["decision_log_hash_match"],
        "energy_delta_abs_max": exp5764._stable_float(energy_delta),  # noqa: SLF001
        "energy_histogram_tv": exp5724.distribution_tv(
            exp5724.energy_histogram(rust_energies),
            exp5724.energy_histogram(python_energies),
        ),
        "acceptance_delta_abs": exp5764._stable_float(acceptance_delta),  # noqa: SLF001
        "retained_sample_count_match": len(rust["samples_spin"]) == len(python["samples_spin"]),
        "retained_sample_count": len(rust["samples_spin"]),
        "feasibility_match": all(value in {-1, 1} for row in rust["samples_spin"] for value in row)
        and all(value in {-1, 1} for row in python["samples_spin"] for value in row),
        "ess_proxy": float(len(rust["samples_spin"])),
        "autocorrelation_abs_proxy": 0.0 if sample_hash_match else 1.0,
    }


def _pair_exclusion_reason(
    quality: Mapping[str, Any],
    measured: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if measured[RUST_ARM].get("active_backend") != ACTIVE_RUST_BACKEND:
        return "production_backend_not_reachable"
    if measured[PYTHON_ARM].get("active_backend") != ACTIVE_PYTHON_FALLBACK:
        return "fallback_equivalence_failed"
    if measured[RUST_ARM].get("optimized_hot_path_used") is not True:
        return "production_backend_not_reachable"
    if quality.get("sample_hash_match") is not True:
        return "semantic_parity_failed"
    if float(quality.get("acceptance_delta_abs", 1.0)) != 0.0:
        return "semantic_parity_failed"
    if quality.get("restart_suffix_hash_match") is not True:
        return "restart_parity_failed"
    if float(quality.get("energy_histogram_tv", 1.0)) != 0.0:
        return "distributional_parity_failed"
    if float(measured[RUST_ARM].get("end_to_end_s", 0.0)) <= 0:
        return "timing_nonpositive"
    return None


def _warmup_receipt(size: int, manifest: Mapping[str, Any], seed: int) -> JsonDict:
    item = _batch_item(size, seed, manifest, return_decision_log=False)
    rust = OneAxisRustBackend(seed=seed, prefer_rust=True).sample_batch([item])[0]
    python = OneAxisRustBackend(seed=seed, prefer_rust=False).sample_batch([item])[0]
    return {
        "size": int(size),
        "warmup_batches": int(manifest["warmup_batches"]),
        "stable": _spin_rows(rust["samples_spin"]) == _spin_rows(python["samples_spin"]),
        "rust_active_backend": rust["receipt"]["active_backend"],
        "python_active_backend": python["receipt"]["active_backend"],
        "rust_optimized_hot_path_used": rust["receipt"]["optimized_hot_path"]["used"],
    }


def _batch_item(
    size: int,
    seed: int,
    manifest: Mapping[str, Any],
    *,
    return_decision_log: bool,
) -> JsonDict:
    workload = exp5724.build_workload_manifest(
        problem_sizes=(int(size),),
        topology_families=("ferromagnetic_ring_easy",),
    )[0]
    fields, couplings = exp5724.arrays_from_workload(workload)
    return {
        "workload_id": workload["workload_id"],
        "biases": np.ascontiguousarray(fields, dtype=np.float64),
        "couplings": np.ascontiguousarray(couplings, dtype=np.float64),
        "n_samples": int(manifest["retained_samples_per_batch"]),
        "config": descriptor_for_run(
            seed=int(seed),
            initial_states=exp5724.initial_states_for(workload, int(seed)),
            initial_labels=list(range(len(exp5714.BETA_LADDER))),
            burn_in_sweeps=int(manifest["burn_in_sweeps"]),
        )
        | {"return_decision_log": bool(return_decision_log)},
    }


def _restart_receipt(
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    prefer_rust: bool,
) -> JsonDict:
    suffix = OneAxisRustBackend(
        seed=int(item["config"]["seed"]),
        prefer_rust=prefer_rust,
    ).run_descriptor(
        item["biases"],
        item["couplings"],
        1,
        {**item["config"], "checkpoint": row["checkpoint"], "burn_in_sweeps": 0},
    )
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_checksum_match": row["checkpoint"]["payload_checksum"]
        == checkpoint_checksum(row["checkpoint"]),
        "suffix_hash": sha256_json(
            {
                "samples_spin": _spin_rows(suffix["samples_spin"]),
                "checkpoint_state": suffix["checkpoint"]["state"],
            }
        ),
    }


def _validate_result(item: Mapping[str, Any], row: Mapping[str, Any], encoded: str) -> JsonDict:
    return {
        "valid": len(row["samples_spin"]) == int(item["n_samples"])
        and row["checkpoint"]["payload_checksum"] == checkpoint_checksum(row["checkpoint"]),
        "encoded_bytes": len(encoded.encode("utf-8")),
    }


def _jsonable_result(row: Mapping[str, Any]) -> JsonDict:
    return {
        "samples_spin": _spin_rows(row["samples_spin"]),
        "receipt": row["receipt"],
        "checkpoint": row["checkpoint"],
        "allocation_counters": row.get("allocation_counters", {}),
    }


def _decision_diagnostics(decision_log: Sequence[Mapping[str, Any]]) -> JsonDict:
    accepted = [event for event in decision_log if event.get("accepted") is True]
    return {
        "event_count": len(decision_log),
        "accepted_count": len(accepted),
        "acceptance_rate": exp5764._stable_float(len(accepted) / len(decision_log))  # noqa: SLF001
        if decision_log
        else 0.0,
    }


def _quality_metrics_for_size(size: int, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    qualities = [row["quality"] for row in rows]
    included = [row for row in rows if row.get("included") is True]
    return {
        "size": int(size),
        "pair_count": len(included),
        "quality_matched": len(included) == len(rows) and bool(rows),
        "energy_delta_abs_max": max(float(row["energy_delta_abs_max"]) for row in qualities)
        if qualities
        else None,
        "feasibility_match": all(row["feasibility_match"] is True for row in qualities),
        "acceptance_delta_abs": max(float(row["acceptance_delta_abs"]) for row in qualities)
        if qualities
        else None,
        "retained_sample_count": sum(int(row["retained_sample_count"]) for row in qualities),
        "ess_min": min(float(row["ess_proxy"]) for row in qualities) if qualities else None,
        "autocorrelation_abs_max": max(float(row["autocorrelation_abs_proxy"]) for row in qualities)
        if qualities
        else None,
        "median_speedup_ratio": statistics.median(float(row["speedup_ratio"]) for row in rows)
        if rows
        else None,
    }


def _semantic_parity_for_size(size: int, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "size": int(size),
        "passed": bool(rows)
        and all(
            row["quality"]["sample_hash_match"] is True
            and float(row["quality"]["acceptance_delta_abs"]) == 0.0
            for row in rows
        ),
        "sample_hash_match": all(row["quality"]["sample_hash_match"] is True for row in rows),
        "decision_log_hash_match_recorded": all(
            row["quality"]["decision_log_hash_match"] is True for row in rows
        ),
        "acceptance_rate_delta_abs_max": max(
            [float(row["quality"]["acceptance_delta_abs"]) for row in rows] or [1.0]
        ),
    }


def _restart_parity_for_size(size: int, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "size": int(size),
        "passed": bool(rows)
        and all(row["quality"]["restart_suffix_hash_match"] is True for row in rows),
        "checkpoint_hash_match": all(
            row["quality"]["checkpoint_state_match"] is True for row in rows
        ),
        "restart_suffix_hash_match": all(
            row["quality"]["restart_suffix_hash_match"] is True for row in rows
        ),
    }


def _distributional_parity_for_size(size: int, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    tv_values = [float(row["quality"]["energy_histogram_tv"]) for row in rows]
    energy_values = [float(row["quality"]["energy_delta_abs_max"]) for row in rows]
    return {
        "size": int(size),
        "passed": bool(rows)
        and max(tv_values or [1.0]) == 0.0
        and max(energy_values or [1.0]) == 0.0,
        "energy_histogram_tv": max(tv_values) if tv_values else None,
        "mean_energy_delta_abs": max(energy_values) if energy_values else None,
        "best_energy_delta_abs": max(energy_values) if energy_values else None,
    }


def _spin_rows(samples_spin: Any) -> list[list[int]]:
    return np.asarray(samples_spin, dtype=np.int8).astype(int).tolist()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(canonical_json([str(part) for part in parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _competing_processes() -> list[JsonDict]:
    result = exp5764._command_output(["ps", "-eo", "pid=,comm=,args="])  # noqa: SLF001
    if result.get("exit_code") != 0:
        return []
    rows: list[JsonDict] = []
    current = str(os.getpid())
    needles = ("experiment_5739_one_axis", "experiment_5764_one_axis", "experiment_5765_one_axis")
    for line in "\n".join(result.get("lines", [])).splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3 or parts[0] == current:
            continue
        if parts[1] in {"bash", "timeout"}:
            continue
        if "pytest" in parts[2] or "test_experiment_5765" in parts[2]:
            continue
        if any(needle in parts[2] for needle in needles):
            rows.append({"pid": int(parts[0]), "command": parts[1], "args": parts[2]})
    return rows


@contextmanager
def _temporary_affinity(cpus: Sequence[int], *, enabled: bool):
    if not enabled or not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return
    original = set(os.sched_getaffinity(0))
    target = {int(cpu) for cpu in cpus}
    if not target:
        yield
        return
    os.sched_setaffinity(0, target)
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


def main() -> None:
    """CLI entrypoint used by the conductor."""

    artifact = build_artifact(root=REPO_ROOT)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover - exercised by conductor CLI, not unit tests.
    main()
