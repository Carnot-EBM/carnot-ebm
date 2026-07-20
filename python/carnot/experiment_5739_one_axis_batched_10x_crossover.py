"""Exp5739 one-axis batched Rust/Python 10x crossover benchmark.

Spec refs: REQ-SAMPLE-5739, SCENARIO-SAMPLE-5739.

This experiment asks one narrow CPU software question: whether the production
reachable Exp5738 Rust/PyO3 ``sample_batch`` path proves a 10x end-to-end
throughput win over the exact Python fallback under identical work and quality.
The benchmark keeps serialization, validation, allocation, checkpoint, PyO3,
and restart costs inside the primary timing because those costs are part of
the production path a caller actually pays.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5724_one_axis_rust_python_matched_crossover as exp5724
from carnot import experiment_5738_one_axis_rust_batched_backend as exp5738
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
RESULT_RELATIVE_PATH = Path("results/experiment_5739_one_axis_batched_10x_crossover.json")

EXPERIMENT = 5739
EXPERIMENT_ID = "exp5739-one-axis-batched-10x-crossover"
MILESTONE = "2026.07.539"
RUN_DATE = "2026-07-20"
SCHEMA = "carnot.experiment_5739.one_axis_batched_10x_crossover.v1"
SPEC_REFS = ("REQ-SAMPLE-5739", "SCENARIO-SAMPLE-5739")
INFERENCE_SUBSTRATE = "matched_cpu_python_rust_batched_sampler_benchmark"
TERMINAL_PREFIXES = ("complete:", "blocked:")

RUST_ARM = "rust_pyo3_batched"
PYTHON_ARM = "python_exact_fallback_batched"
ARMS = (RUST_ARM, PYTHON_ARM)

DEFAULT_PROBLEM_SIZES = (48, 96, 192)
DEFAULT_TOPOLOGY_FAMILIES = exp5724.DEFAULT_TOPOLOGY_FAMILIES
DEFAULT_BATCH_SIZES = (1, 4)
DEFAULT_RANDOM_SEEDS = tuple(range(5_739_000, 5_739_030))
DEFAULT_WARMUP_COUNT = 1
DEFAULT_MEASURED_BATCH_COUNT = 30
DEFAULT_SAMPLE_SWEEPS = 2
DEFAULT_BURN_IN_SWEEPS = 1
DEFAULT_BENCHMARK_ORDER_SEED = 5_739_020
BOOTSTRAP_RESAMPLES = 500
FAMILYWISE_ALPHA = 0.05

THREAD_ENV_KEYS = exp5724.THREAD_ENV_KEYS
PHASES = ("setup", "sample_batch", "serialization", "validation", "restart")

QUALITY_MARGINS: JsonDict = {
    "energy_histogram_tv_max": 0.0,
    "best_energy_delta_abs_max": 1e-12,
    "mean_energy_delta_abs_max": 1e-12,
    "sample_count_match_required": True,
    "work_counters_match_required": True,
    "restart_match_required": True,
    "result_order_match_required": True,
    "rust_active_backend_required": ACTIVE_RUST_BACKEND,
    "python_active_backend_required": ACTIVE_PYTHON_FALLBACK,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Names the artifact contract so downstream validators do not infer fields from experiment number alone.",
    "experiment": "Keeps the numeric experiment identifier explicit for corpus indexing without turning it into a metric.",
    "experiment_id": "Provides a stable human-readable artifact identity for cross-run provenance.",
    "milestone": "Records the conductor milestone without letting milestone status substitute for benchmark evidence.",
    "run_date": "Anchors the local CPU software measurement to the requested 2026-07-20 run date.",
    "spec_refs": "Binds the artifact to REQ-SAMPLE-5739 and SCENARIO-SAMPLE-5739.",
    "duration_s": "Records real wall-clock artifact construction time for fabrication and reproducibility review.",
    "field_principles": "Explains why every Exp5739 field exists before a reviewer trusts the JSON shape.",
    "preconditions_checked": "Shows every upstream, build, topology, affinity, thread, and workload gate checked before timing interpretation.",
    "upstream_gate_receipts": "Pins Exp5724 null timing evidence and Exp5738 batched backend readiness plus source hashes.",
    "software_receipt": "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay the two production arms.",
    "build_profile": "Freezes release/debug, PyO3 ABI, bulk-run symbol, and feature state before any 10x timing claim.",
    "cpu_topology": "Separates physical and logical CPU evidence from accelerator or board claims.",
    "thread_receipts": "Records matched thread policy for both arms in each regime and discloses implementation asymmetry.",
    "cpu_affinity": "Shows the one-core and fixed-core placements used for timing.",
    "preregistered_protocol": "Freezes workloads, sizes, batches, seeds, ladders, budgets, checkpoints, warmups, repetitions, quality margins, and 10x null rule.",
    "workload_manifest": "Lists every size and topology Hamiltonian shared by Python and Rust batch arms.",
    "arm_configs": "Names the Rust PyO3 batch and exact Python fallback batch configurations under one production API.",
    "batch_sizes": "Makes the amount of independent work per batch explicit instead of implying unmeasured amortization.",
    "problem_sizes": "Makes the 48/96/192 large-size panel explicit, including any infeasible-size blocker.",
    "random_seeds": "Records the independent batch seed schedule for exact replay.",
    "warmup_count": "Prevents cold-start effects from entering primary measurements.",
    "measured_batch_count": "Proves every qualified cell has at least thirty independent measured batches.",
    "matched_work_receipts": "Proves identical work, energy accounting, samples, transitions, checkpoints, and restart behavior.",
    "quality_metrics_by_pair": "Reports matched quality before any speedup ratio enters a distribution.",
    "quality_matched_pair_count": "Counts only pairs that passed quality gates before timing intervals.",
    "excluded_pair_reasons": "Keeps mismatches visible instead of silently relabeling them as speed evidence.",
    "end_to_end_times": "Preserves primary latency evidence including serialization, validation, PyO3, allocation, checkpoint, and restart costs.",
    "throughput_distributions": "Reports samples-per-second distributions from the same end-to-end measurements.",
    "phase_times": "Preserves bottleneck evidence without subtracting phases from primary timing.",
    "peak_rss_by_arm": "Records memory pressure by arm and thread regime so speed is not traded for hidden RSS growth.",
    "paired_speedup_ratios": "Reports paired Python/Rust ratios on identical batches rather than unrelated aggregate means.",
    "paired_speedup_intervals": "Uses adjusted confidence intervals so a 10x claim cannot rest on noisy multiple comparisons.",
    "qualified_10x_sizes": "Lists only consecutive larger sizes whose adjusted lower confidence bound is at least 10.0.",
    "qualified_10x_thread_regime": "Names the single thread regime, if any, where the strict consecutive-size 10x rule passed.",
    "rust_batched_10x_ready_score": "Equals 1.0 only when matched quality and adjusted intervals prove the strict 10x CPU software rule.",
    "timing_claimed": "Bare true declares this is a timing benchmark, unlike Exp5738 readiness-only evidence.",
    "software_speedup_claimed": "Bare true is allowed only when the strict 10x CPU software rule passes.",
    "gpu_speedup_claimed": "Bare false prevents this CPU benchmark from reopening GPU claims.",
    "hardware_speedup_claimed": "Bare false prevents local CPU software timing from becoming a board or TSU claim.",
    "fpga_or_tsu_used": "Bare false records that no FPGA, TSU, or board participated.",
    "inference_substrate": "Declares matched CPU Python/Rust batched one-axis sampler benchmarking, not LLM inference or accelerator timing.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether the strict 10x rule passed or terminally failed.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable hashes and receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content with Carnot's SHA-256 convention."""

    return exp5724.sha256_json(value)


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for source and upstream receipts."""

    return exp5724.file_sha256(path)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def preregistered_protocol(
    *,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_batch_count: int = DEFAULT_MEASURED_BATCH_COUNT,
    sample_sweeps: int = DEFAULT_SAMPLE_SWEEPS,
    burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS,
    allow_underpowered: bool = False,
) -> JsonDict:
    """Freeze the benchmark design before any speed ratio can be interpreted."""

    sizes = tuple(int(size) for size in problem_sizes)
    families = tuple(str(family) for family in topology_families)
    batches = tuple(int(batch) for batch in batch_sizes)
    seeds = tuple(int(seed) for seed in random_seeds)
    if not allow_underpowered:
        if not {48, 96, 192}.issubset(set(sizes)):
            raise ValueError("problem_sizes must include 48, 96, and 192")
        if len(families) < 3:
            raise ValueError("topology_families must contain at least three families")
        if len(seeds) < 30:
            raise ValueError("random_seeds must contain at least thirty seeds")
        if int(measured_batch_count) < 30:
            raise ValueError("measured_batch_count must be at least thirty")
    if any(size <= 0 for size in sizes):
        raise ValueError("problem_sizes must be positive")
    if any(batch <= 0 for batch in batches):
        raise ValueError("batch_sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("problem_sizes must be unique")
    if len(set(families)) != len(families):
        raise ValueError("topology_families must be unique")
    if len(set(batches)) != len(batches):
        raise ValueError("batch_sizes must be unique")
    if not seeds:
        raise ValueError("random_seeds must not be empty")
    if int(warmup_count) < 0:
        raise ValueError("warmup_count must be nonnegative")
    if int(sample_sweeps) <= 0 or int(burn_in_sweeps) < 0:
        raise ValueError("transition budgets must be positive samples and nonnegative burn-in")

    return {
        "schema": "carnot.exp5739.preregistered_protocol.v1",
        "frozen_before_timing": True,
        "run_date": RUN_DATE,
        "problem_sizes": list(sizes),
        "topology_families": list(families),
        "batch_sizes": list(batches),
        "random_seeds": list(seeds),
        "benchmark_order_seed": DEFAULT_BENCHMARK_ORDER_SEED,
        "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
        "proposal_std": float(exp5714.exp5622.CDLS_PROPOSAL_STD),
        "drift_scale": float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        "energy_convention": ENERGY_CONVENTION,
        "algorithm": ONE_AXIS_ALGORITHM,
        "topology": ONE_AXIS_TOPOLOGY,
        "warmup_count": int(warmup_count),
        "measured_batch_count": int(measured_batch_count),
        "sample_sweeps": int(sample_sweeps),
        "burn_in_sweeps": int(burn_in_sweeps),
        "checkpoint_restarts_per_item": 1,
        "stopping_rule": "fixed_sweep_budget_no_early_stop",
        "quality_margins": dict(QUALITY_MARGINS),
        "thread_regime_rule": "one physical-core placement and fixed recorded core allocation",
        "seed_schedule": (
            "batch base seed is random_seeds[batch_index]; item seeds are derived from "
            "workload id, batch size, batch index, and item index"
        ),
        "primary_timing": "end_to_end_time_no_overhead_subtraction",
        "phase_costs_included_in_primary": list(PHASES),
        "speedup_ratio": "python_end_to_end_time / rust_end_to_end_time",
        "multiple_comparison_correction": "bonferroni_over_thread_regime_size_intervals",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "claim_rule": (
            "rust_batched_10x_ready_score is 1.0 only when matched quality holds and "
            "the adjusted lower confidence bound is >=10.0 at two consecutive larger "
            "sizes in the same thread regime"
        ),
        "gpu_fpga_tsu_hardware_claims_allowed": False,
    }


def build_workload_manifest(
    *,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
) -> list[JsonDict]:
    """Build deterministic Ising workloads reused by both batched arms."""

    return exp5724.build_workload_manifest(
        problem_sizes=tuple(int(size) for size in problem_sizes),
        topology_families=tuple(str(family) for family in topology_families),
    )


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    benchmark_runner: BenchmarkRunner | None = None,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_batch_count: int = DEFAULT_MEASURED_BATCH_COUNT,
    freeze_affinity: bool = True,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build and validate the terminal Exp5739 artifact."""

    del tests_added_or_reused
    started = time.perf_counter()
    root_path = Path(root)
    protocol = preregistered_protocol(
        problem_sizes=problem_sizes,
        topology_families=topology_families,
        batch_sizes=batch_sizes,
        random_seeds=random_seeds,
        warmup_count=warmup_count,
        measured_batch_count=measured_batch_count,
    )
    workloads = build_workload_manifest(
        problem_sizes=protocol["problem_sizes"],
        topology_families=protocol["topology_families"],
    )
    affinity = cpu_affinity_receipt(freeze_affinity=freeze_affinity)
    thread_regimes = list(affinity["thread_regimes"])
    upstream = upstream_gate_receipts(root_path)
    runner = benchmark_runner or run_matched_batched_benchmark
    if upstream.get("exp5738", {}).get("ready") is True:
        evidence = runner(protocol=protocol, workloads=workloads, thread_regimes=thread_regimes)
    else:
        evidence = blocked_benchmark_evidence(
            protocol=protocol,
            workloads=workloads,
            thread_regimes=thread_regimes,
            reason="upstream_exp5738_not_ready",
        )
    qualified_sizes, qualified_regime = qualified_10x_from_intervals(
        evidence["paired_speedup_intervals"],
        problem_sizes=protocol["problem_sizes"],
    )
    quality_count = sum(
        1 for row in evidence["quality_metrics_by_pair"] if row.get("quality_matched") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions_checked(root_path, upstream, affinity, protocol),
        "upstream_gate_receipts": upstream,
        "software_receipt": software_receipt(root_path),
        "build_profile": build_profile(root_path),
        "cpu_topology": cpu_topology(affinity),
        "thread_receipts": thread_receipts(thread_regimes),
        "cpu_affinity": affinity,
        "preregistered_protocol": protocol,
        "workload_manifest": workloads,
        "arm_configs": arm_configs(),
        "batch_sizes": list(protocol["batch_sizes"]),
        "problem_sizes": list(protocol["problem_sizes"]),
        "random_seeds": list(protocol["random_seeds"]),
        "warmup_count": int(protocol["warmup_count"]),
        "measured_batch_count": int(protocol["measured_batch_count"]),
        "matched_work_receipts": evidence["matched_work_receipts"],
        "quality_metrics_by_pair": evidence["quality_metrics_by_pair"],
        "quality_matched_pair_count": quality_count,
        "excluded_pair_reasons": excluded_pair_reasons(evidence["quality_metrics_by_pair"]),
        "end_to_end_times": evidence["end_to_end_times"],
        "throughput_distributions": evidence["throughput_distributions"],
        "phase_times": evidence["phase_times"],
        "peak_rss_by_arm": evidence["peak_rss_by_arm"],
        "paired_speedup_ratios": evidence["paired_speedup_ratios"],
        "paired_speedup_intervals": evidence["paired_speedup_intervals"],
        "qualified_10x_sizes": qualified_sizes,
        "qualified_10x_thread_regime": qualified_regime,
        "rust_batched_10x_ready_score": 0.0,
        "timing_claimed": True,
        "software_speedup_claimed": bool(qualified_sizes),
        "gpu_speedup_claimed": False,
        "hardware_speedup_claimed": False,
        "fpga_or_tsu_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: 10x gates not evaluated",
    }
    artifact["rust_batched_10x_ready_score"] = rust_batched_10x_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_matched_batched_benchmark(
    *,
    protocol: Mapping[str, Any],
    workloads: Sequence[Mapping[str, Any]],
    thread_regimes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Run real Rust/PyO3 and exact-Python batches on identical inputs."""

    timing_records: list[JsonDict] = []
    quality_rows: list[JsonDict] = []
    work_rows: list[JsonDict] = []
    for regime in thread_regimes:
        regime_id = str(regime["regime_id"])
        with _temporary_affinity(regime.get("cpus", []), bool(regime.get("affinity_enforced"))):
            for workload in workloads:
                for batch_size in protocol["batch_sizes"]:
                    _warm_cell(protocol, workload, int(batch_size))
                    for batch_index in range(int(protocol["measured_batch_count"])):
                        items = batch_items_for(protocol, workload, int(batch_size), batch_index)
                        measured: dict[str, JsonDict] = {}
                        for arm in _arm_order(regime_id, workload["workload_id"], batch_index):
                            record = measure_batch_arm(
                                protocol=protocol,
                                workload=workload,
                                batch_size=int(batch_size),
                                batch_index=batch_index,
                                thread_regime=regime_id,
                                arm=arm,
                                items=items,
                            )
                            measured[arm] = record
                            timing_records.append(record)
                        quality, work = quality_and_work_for_pair(
                            protocol=protocol,
                            workload=workload,
                            batch_size=int(batch_size),
                            batch_index=batch_index,
                            thread_regime=regime_id,
                            rust_record=measured[RUST_ARM],
                            python_record=measured[PYTHON_ARM],
                        )
                        quality_rows.append(quality)
                        work_rows.append(work)

    ratio_rows = paired_speedup_ratios_from_records(
        timing_records,
        quality_rows=quality_rows,
    )
    intervals = paired_speedup_intervals_from_ratios(
        ratio_rows,
        quality_rows=quality_rows,
        thread_regimes=[str(row["regime_id"]) for row in thread_regimes],
        problem_sizes=protocol["problem_sizes"],
        measured_batch_count=int(protocol["measured_batch_count"]),
    )
    return {
        "matched_work_receipts": work_rows,
        "quality_metrics_by_pair": quality_rows,
        "excluded_pair_reasons": excluded_pair_reasons(quality_rows),
        "end_to_end_times": summarize_end_to_end(timing_records),
        "throughput_distributions": summarize_throughput(timing_records),
        "phase_times": summarize_phase_times(timing_records),
        "peak_rss_by_arm": peak_rss_by_arm(timing_records),
        "paired_speedup_ratios": ratio_rows,
        "paired_speedup_intervals": intervals,
    }


def blocked_benchmark_evidence(
    *,
    protocol: Mapping[str, Any],
    workloads: Sequence[Mapping[str, Any]],
    thread_regimes: Sequence[Mapping[str, Any]],
    reason: str,
) -> JsonDict:
    """Preserve denominators when an upstream gate blocks benchmark timing."""

    quality: list[JsonDict] = []
    work: list[JsonDict] = []
    for regime in thread_regimes:
        regime_id = str(regime["regime_id"])
        for workload in workloads:
            for batch_size in protocol["batch_sizes"]:
                cell = cell_id(regime_id, workload, int(batch_size))
                work.append(
                    {
                        "cell_id": cell,
                        "thread_regime": regime_id,
                        "size": int(workload["size"]),
                        "family": workload["family"],
                        "batch_size": int(batch_size),
                        "measured_batch_count": 0,
                        "matched": False,
                        "excluded_reason": reason,
                    }
                )
                for batch_index in range(int(protocol["measured_batch_count"])):
                    quality.append(
                        {
                            "pair_id": f"{cell}:batch{batch_index}",
                            "cell_id": cell,
                            "thread_regime": regime_id,
                            "size": int(workload["size"]),
                            "family": workload["family"],
                            "batch_size": int(batch_size),
                            "batch_index": batch_index,
                            "quality_matched": False,
                            "excluded_reason": reason,
                        }
                    )
    intervals = [
        {
            "thread_regime": str(regime["regime_id"]),
            "size": int(size),
            "comparison_count": len(thread_regimes) * len(protocol["problem_sizes"]),
            "adjusted_alpha": round(FAMILYWISE_ALPHA / max(1, len(thread_regimes) * len(protocol["problem_sizes"])), 12),
            "repetition_count": 0,
            "lower_confidence_bound": None,
            "upper_confidence_bound": None,
            "passes_10x_lower_bound": False,
            "quality_matched": False,
            "excluded_reason": reason,
        }
        for regime in thread_regimes
        for size in protocol["problem_sizes"]
    ]
    return {
        "matched_work_receipts": work,
        "quality_metrics_by_pair": quality,
        "excluded_pair_reasons": excluded_pair_reasons(quality),
        "end_to_end_times": [],
        "throughput_distributions": [],
        "phase_times": [],
        "peak_rss_by_arm": {},
        "paired_speedup_ratios": [],
        "paired_speedup_intervals": intervals,
    }


def measure_batch_arm(
    *,
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    batch_size: int,
    batch_index: int,
    thread_regime: str,
    arm: str,
    items: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Measure one arm end-to-end, including validation and restart costs."""

    del protocol
    prefer_rust = arm == RUST_ARM
    phase_samples: dict[str, float] = {}
    start = time.perf_counter()

    phase_start = time.perf_counter()
    backend = OneAxisRustBackend(seed=int(items[0]["config"]["seed"]), prefer_rust=prefer_rust)
    phase_samples["setup"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    rows = backend.sample_batch(items)
    phase_samples["sample_batch"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    encoded = canonical_json(
        [
            {
                "workload_id": row["workload_id"],
                "receipt": row["receipt"],
                "checkpoint": row["checkpoint"],
                "samples_spin": row["samples_spin"],
            }
            for row in rows
        ]
    )
    phase_samples["serialization"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    validation = validate_batch_rows(items, rows, encoded)
    phase_samples["validation"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    restart_receipt = restart_receipt_for_rows(items, rows, prefer_rust=prefer_rust)
    phase_samples["restart"] = time.perf_counter() - phase_start

    elapsed = time.perf_counter() - start
    samples_per_batch = int(batch_size) * int(items[0]["n_samples"])
    return {
        "pair_id": f"{cell_id(thread_regime, workload, batch_size)}:batch{batch_index}",
        "cell_id": cell_id(thread_regime, workload, batch_size),
        "thread_regime": thread_regime,
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "batch_size": int(batch_size),
        "batch_index": int(batch_index),
        "arm": arm,
        "active_backends": [row["receipt"]["active_backend"] for row in rows],
        "end_to_end_s": _stable_float(elapsed),
        "phase_s": {key: _stable_float(value) for key, value in phase_samples.items()},
        "throughput_samples_per_s": _stable_float(samples_per_batch / elapsed),
        "samples_per_batch": samples_per_batch,
        "peak_rss_kib": current_peak_rss_kib(),
        "validation": validation,
        "restart_receipt": restart_receipt,
        "rows": rows,
        "items": list(items),
    }


def validate_batch_rows(
    items: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    encoded: str,
) -> JsonDict:
    """Validate ordering, sample counts, and checkpoint checksums for a batch."""

    order_match = [str(item["workload_id"]) for item in items] == [
        str(row.get("workload_id")) for row in rows
    ]
    sample_count_match = all(
        len(row.get("samples_spin", [])) == int(item["n_samples"])
        for item, row in zip(items, rows, strict=False)
    ) and len(items) == len(rows)
    checkpoint_match = all(
        row.get("checkpoint", {}).get("payload_checksum")
        == checkpoint_checksum(row.get("checkpoint", {}))
        for row in rows
    )
    return {
        "valid": bool(order_match and sample_count_match and checkpoint_match),
        "result_order_match": bool(order_match),
        "sample_count_match": bool(sample_count_match),
        "checkpoint_checksum_match": bool(checkpoint_match),
        "encoded_batch_bytes": len(encoded.encode("utf-8")),
    }


def restart_receipt_for_rows(
    items: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    prefer_rust: bool,
) -> JsonDict:
    """Run one checkpoint suffix per item so restart cost stays in timing."""

    suffix_hashes: list[str] = []
    for item, row in zip(items, rows, strict=True):
        suffix = OneAxisRustBackend(
            seed=int(item["config"]["seed"]),
            prefer_rust=prefer_rust,
        ).run_descriptor(
            item["biases"],
            item["couplings"],
            1,
            {**item["config"], "checkpoint": row["checkpoint"], "burn_in_sweeps": 0},
        )
        suffix_hashes.append(
            sha256_json(
                {
                    "samples_spin": suffix["samples_spin"],
                    "decision_log": suffix["decision_log"],
                    "checkpoint_state": suffix["checkpoint"]["state"],
                }
            )
        )
    return {
        "restart_count": len(suffix_hashes),
        "suffix_hash": sha256_json(suffix_hashes),
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
    }


def quality_and_work_for_pair(
    *,
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    batch_size: int,
    batch_index: int,
    thread_regime: str,
    rust_record: Mapping[str, Any],
    python_record: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    """Compare matched quality and work before a pair may enter speed ratios."""

    del protocol
    rust_metrics = batch_quality_metrics(rust_record)
    python_metrics = batch_quality_metrics(python_record)
    work = matched_work_receipt(workload, batch_size, batch_index, thread_regime, rust_record, python_record)
    energy_tv = exp5724.distribution_tv(
        rust_metrics["energy_histogram"],
        python_metrics["energy_histogram"],
    )
    row: JsonDict = {
        "pair_id": rust_record["pair_id"],
        "cell_id": rust_record["cell_id"],
        "thread_regime": thread_regime,
        "size": int(workload["size"]),
        "family": workload["family"],
        "batch_size": int(batch_size),
        "batch_index": int(batch_index),
        "rust_active_backends": list(rust_record["active_backends"]),
        "python_active_backends": list(python_record["active_backends"]),
        "sample_count_match": rust_record["validation"]["sample_count_match"]
        and python_record["validation"]["sample_count_match"]
        and rust_metrics["sample_count"] == python_metrics["sample_count"],
        "work_counters_match": work["matched"],
        "restart_match": rust_record["restart_receipt"]["suffix_hash"]
        == python_record["restart_receipt"]["suffix_hash"],
        "result_order_match": rust_record["validation"]["result_order_match"]
        and python_record["validation"]["result_order_match"],
        "energy_histogram_tv": energy_tv,
        "best_energy_delta_abs": abs(rust_metrics["best_energy"] - python_metrics["best_energy"]),
        "mean_energy_delta_abs": abs(rust_metrics["mean_energy"] - python_metrics["mean_energy"]),
        "rust_metrics": rust_metrics,
        "python_metrics": python_metrics,
        "quality_matched": False,
        "excluded_reason": None,
    }
    row["quality_matched"] = quality_matched(row)
    if row["quality_matched"] is not True:
        row["excluded_reason"] = quality_exclusion_reason(row)
    return row, work


def batch_quality_metrics(record: Mapping[str, Any]) -> JsonDict:
    """Summarize exact energy and sample hashes across every batch item."""

    energies: list[float] = []
    sample_count = 0
    sample_hash_inputs: list[Any] = []
    decision_hash_inputs: list[Any] = []
    for item, row in zip(_items_from_record(record), record["rows"], strict=True):
        fields = np.asarray(item["biases"], dtype=np.float64)
        couplings = np.asarray(item["couplings"], dtype=np.float64)
        for sample in row["samples_spin"]:
            energies.append(exp5724.ising_energy(fields, couplings, sample))
        sample_count += len(row["samples_spin"])
        sample_hash_inputs.append(row["samples_spin"])
        decision_hash_inputs.append(row["decision_log"])
    return {
        "sample_count": sample_count,
        "best_energy": min(energies) if energies else 0.0,
        "mean_energy": statistics.fmean(energies) if energies else 0.0,
        "energy_histogram": exp5724.energy_histogram(energies),
        "sample_hash": sha256_json(sample_hash_inputs),
        "decision_log_hash": sha256_json(decision_hash_inputs),
    }


def matched_work_receipt(
    workload: Mapping[str, Any],
    batch_size: int,
    batch_index: int,
    thread_regime: str,
    rust_record: Mapping[str, Any],
    python_record: Mapping[str, Any],
) -> JsonDict:
    """Prove both arms used identical transition, sample, and restart work."""

    rust = batch_work_counters(rust_record)
    python = batch_work_counters(python_record)
    return {
        "pair_id": rust_record["pair_id"],
        "cell_id": rust_record["cell_id"],
        "thread_regime": thread_regime,
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "batch_size": int(batch_size),
        "batch_index": int(batch_index),
        "matched": rust == python,
        "rust": rust,
        "python": python,
        "energy_convention": ENERGY_CONVENTION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
    }


def batch_work_counters(record: Mapping[str, Any]) -> JsonDict:
    """Aggregate production transition-budget receipts across a batch."""

    corrected = 0
    swaps = 0
    cold_samples = 0
    total_sweeps = 0
    for row in record["rows"]:
        budget = row["receipt"]["transition_budget"]
        corrected += int(budget["corrected_transitions"])
        swaps += int(budget["swap_attempts"])
        cold_samples += int(budget["cold_target_samples"])
        total_sweeps += int(budget["total_sweeps"])
    return {
        "batch_size": len(record["rows"]),
        "corrected_transitions": corrected,
        "swap_attempts": swaps,
        "energy_evaluations": 2 * (corrected + swaps),
        "cold_target_samples": cold_samples,
        "total_sweeps": total_sweeps,
        "checkpoint_restarts": int(record["restart_receipt"]["restart_count"]),
        "stopping_rule": "fixed_sweep_budget_no_early_stop",
    }


def quality_matched(row: Mapping[str, Any]) -> bool:
    """Apply the preregistered quality gate for one paired batch."""

    return bool(
        all(backend == ACTIVE_RUST_BACKEND for backend in row.get("rust_active_backends", []))
        and all(
            backend == ACTIVE_PYTHON_FALLBACK
            for backend in row.get("python_active_backends", [])
        )
        and row.get("sample_count_match") is True
        and row.get("work_counters_match") is True
        and row.get("restart_match") is True
        and row.get("result_order_match") is True
        and float(row.get("energy_histogram_tv", 1.0))
        <= QUALITY_MARGINS["energy_histogram_tv_max"]
        and float(row.get("best_energy_delta_abs", 1.0))
        <= QUALITY_MARGINS["best_energy_delta_abs_max"]
        and float(row.get("mean_energy_delta_abs", 1.0))
        <= QUALITY_MARGINS["mean_energy_delta_abs_max"]
    )


def quality_exclusion_reason(row: Mapping[str, Any]) -> str:
    """Return the first preregistered reason a pair cannot enter speedups."""

    if any(backend != ACTIVE_RUST_BACKEND for backend in row.get("rust_active_backends", [])):
        return "rust_arm_not_active_pyo3_batch"
    if any(
        backend != ACTIVE_PYTHON_FALLBACK for backend in row.get("python_active_backends", [])
    ):
        return "python_arm_not_exact_fallback"
    for key in ("sample_count_match", "work_counters_match", "restart_match", "result_order_match"):
        if row.get(key) is not True:
            return key
    for key, margin in (
        ("energy_histogram_tv", "energy_histogram_tv_max"),
        ("best_energy_delta_abs", "best_energy_delta_abs_max"),
        ("mean_energy_delta_abs", "mean_energy_delta_abs_max"),
    ):
        if float(row.get(key, 1.0)) > QUALITY_MARGINS[margin]:
            return key
    return "unknown_quality_mismatch"


def paired_speedup_ratios_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compute paired Python/Rust end-to-end ratios for quality-matched batches."""

    quality_ok = {
        str(row["pair_id"]): row.get("quality_matched") is True for row in quality_rows
    }
    paired: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if quality_ok.get(str(record["pair_id"])) is not True:
            continue
        paired[str(record["pair_id"])][str(record["arm"])] = record
    by_cell: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, Mapping[str, Any]] = {}
    for pair_id, arms in paired.items():
        if RUST_ARM not in arms or PYTHON_ARM not in arms:
            continue
        rust_time = float(arms[RUST_ARM]["end_to_end_s"])
        python_time = float(arms[PYTHON_ARM]["end_to_end_s"])
        if rust_time <= 0 or python_time <= 0:
            continue
        by_cell[str(arms[RUST_ARM]["cell_id"])].append(python_time / rust_time)
        metadata[str(arms[RUST_ARM]["cell_id"])] = arms[RUST_ARM]
        del pair_id
    rows: list[JsonDict] = []
    for cell, ratios in sorted(by_cell.items()):
        first = metadata[cell]
        rows.append(
            {
                "cell_id": cell,
                "thread_regime": first["thread_regime"],
                "size": int(first["size"]),
                "family": first["family"],
                "batch_size": int(first["batch_size"]),
                "repetition_count": len(ratios),
                "ratio_samples": [_stable_float(value) for value in ratios],
                "ratio_mean": _stable_float(statistics.fmean(ratios)),
                "ratio_median": _stable_float(statistics.median(ratios)),
                "rust_faster_fraction": _stable_float(
                    sum(value > 1.0 for value in ratios) / len(ratios)
                ),
            }
        )
    return rows


def paired_speedup_intervals_from_ratios(
    ratio_rows: Sequence[Mapping[str, Any]],
    *,
    quality_rows: Sequence[Mapping[str, Any]],
    thread_regimes: Sequence[str],
    problem_sizes: Sequence[int],
    measured_batch_count: int,
) -> list[JsonDict]:
    """Aggregate ratios by thread regime and size with Bonferroni intervals."""

    comparison_count = max(1, len(tuple(thread_regimes)) * len(tuple(problem_sizes)))
    adjusted_alpha = FAMILYWISE_ALPHA / comparison_count
    quality_by_cell = quality_cells_pass(quality_rows, measured_batch_count=measured_batch_count)
    rows: list[JsonDict] = []
    for regime in thread_regimes:
        for size in problem_sizes:
            samples: list[float] = []
            family_count: set[str] = set()
            batch_size_count: set[int] = set()
            for ratio_row in ratio_rows:
                if str(ratio_row["thread_regime"]) == str(regime) and int(ratio_row["size"]) == int(size):
                    if quality_by_cell.get(str(ratio_row["cell_id"])) is True:
                        samples.extend(float(value) for value in ratio_row["ratio_samples"])
                        family_count.add(str(ratio_row["family"]))
                        batch_size_count.add(int(ratio_row["batch_size"]))
            interval = bootstrap_interval(
                samples,
                seed=_stable_seed64("exp5739-bootstrap", regime, int(size), len(samples)),
                alpha=adjusted_alpha,
            )
            lower = interval[0]
            upper = interval[1]
            rows.append(
                {
                    "thread_regime": str(regime),
                    "size": int(size),
                    "comparison_count": comparison_count,
                    "adjusted_alpha": round(adjusted_alpha, 12),
                    "repetition_count": len(samples),
                    "family_count": len(family_count),
                    "batch_size_count": len(batch_size_count),
                    "lower_confidence_bound": lower,
                    "upper_confidence_bound": upper,
                    "interval": interval,
                    "passes_10x_lower_bound": bool(lower is not None and float(lower) >= 10.0),
                    "quality_matched": bool(samples),
                }
            )
    return rows


def bootstrap_interval(values: Sequence[float], *, seed: int, alpha: float) -> list[float | None]:
    """Return a deterministic bootstrap interval for paired speedup means."""

    vals = np.asarray([float(value) for value in values], dtype=np.float64)
    if vals.size == 0:
        return [None, None]
    if vals.size == 1:
        value = _stable_float(vals[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        means[index] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return [_stable_float(low), _stable_float(high)]


def quality_cells_pass(
    quality_rows: Sequence[Mapping[str, Any]],
    *,
    measured_batch_count: int,
) -> dict[str, bool]:
    """Return whether every measured batch in each cell passed quality."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        groups[str(row["cell_id"])].append(row)
    return {
        cell: len(rows) >= int(measured_batch_count)
        and all(row.get("quality_matched") is True for row in rows)
        for cell, rows in groups.items()
    }


def qualified_10x_from_intervals(
    intervals: Sequence[Mapping[str, Any]],
    *,
    problem_sizes: Sequence[int],
) -> tuple[list[int], str | None]:
    """Return the first thread regime with two consecutive larger 10x sizes."""

    ordered_sizes = [int(size) for size in problem_sizes]
    larger_sizes = ordered_sizes[1:] if len(ordered_sizes) >= 3 else ordered_sizes
    regimes = sorted({str(row["thread_regime"]) for row in intervals})
    by_key = {(str(row["thread_regime"]), int(row["size"])): row for row in intervals}
    for regime in regimes:
        for left, right in zip(larger_sizes, larger_sizes[1:], strict=False):
            left_row = by_key.get((regime, left), {})
            right_row = by_key.get((regime, right), {})
            if (
                left_row.get("quality_matched") is True
                and right_row.get("quality_matched") is True
                and left_row.get("passes_10x_lower_bound") is True
                and right_row.get("passes_10x_lower_bound") is True
            ):
                return [left, right], regime
    return [], None


def rust_batched_10x_ready_score(payload: Mapping[str, Any]) -> float:
    """Return the scalar strict 10x readiness gate."""

    expected_quality = (
        len(payload.get("cpu_affinity", {}).get("thread_regimes", []))
        * len(payload.get("workload_manifest", []))
        * len(payload.get("batch_sizes", []))
        * int(payload.get("measured_batch_count", 0))
    )
    upstream = payload.get("upstream_gate_receipts", {})
    gates = [
        isinstance(upstream, Mapping) and upstream.get("exp5738", {}).get("ready") is True,
        set(payload.get("qualified_10x_sizes", [])) != set(),
        payload.get("qualified_10x_thread_regime") is not None,
        payload.get("quality_matched_pair_count") == expected_quality,
        not payload.get("excluded_pair_reasons"),
        all(row.get("matched") is True for row in payload.get("matched_work_receipts", [])),
        int(payload.get("measured_batch_count", 0)) >= 30,
        {48, 96, 192}.issubset(set(payload.get("problem_sizes", []))),
        payload.get("timing_claimed") is True,
        payload.get("software_speedup_claimed") is True,
        payload.get("gpu_speedup_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("fpga_or_tsu_used") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal complete or blocked verdict."""

    upstream = payload.get("upstream_gate_receipts", {})
    if not isinstance(upstream, Mapping) or upstream.get("exp5738", {}).get("ready") is not True:
        return "blocked: upstream Exp5738 batched backend or backend/build hash gate failed"
    if rust_batched_10x_ready_score(payload) == 1.0:
        return (
            "complete: matched-quality Rust/PyO3 batched CPU software 10x rule passed "
            f"at sizes {payload.get('qualified_10x_sizes')} in "
            f"{payload.get('qualified_10x_thread_regime')}; no GPU, FPGA, TSU, or hardware claim"
        )
    return (
        "complete: terminal null; matched batched Rust/Python CPU evidence did not prove "
        "the strict consecutive larger-size 10x lower-bound rule"
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5739 fields and fail closed on unsafe claim edits."""

    if set(payload.keys()) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("artifact fields mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if not {48, 96, 192}.issubset(set(payload.get("problem_sizes", []))):
        raise ValueError("problem_sizes must include 48, 96, and 192")
    if len(payload.get("random_seeds", [])) < 30:
        raise ValueError("random_seeds must contain at least thirty seeds")
    if int(payload.get("measured_batch_count", 0)) < 30:
        raise ValueError("measured_batch_count must be at least thirty")
    expected_quality = (
        len(payload.get("cpu_affinity", {}).get("thread_regimes", []))
        * len(payload.get("workload_manifest", []))
        * len(payload.get("batch_sizes", []))
        * int(payload.get("measured_batch_count", 0))
    )
    quality_rows = payload.get("quality_metrics_by_pair", [])
    if len(quality_rows) != expected_quality:
        raise ValueError("quality_metrics_by_pair denominator mismatch")
    matched_count = sum(1 for row in quality_rows if row.get("quality_matched") is True)
    if payload.get("quality_matched_pair_count") != matched_count:
        raise ValueError("quality_matched_pair_count mismatch")
    if payload.get("excluded_pair_reasons") != excluded_pair_reasons(quality_rows):
        raise ValueError("excluded_pair_reasons mismatch")
    if payload.get("software_speedup_claimed") is not bool(payload.get("qualified_10x_sizes")):
        raise ValueError("software_speedup_claimed mismatch")
    if payload.get("timing_claimed") is not True:
        raise ValueError("timing_claimed must be true")
    if payload.get("gpu_speedup_claimed") is not False:
        raise ValueError("gpu_speedup_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("fpga_or_tsu_used") is not False:
        raise ValueError("fpga_or_tsu_used must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("rust_batched_10x_ready_score") != rust_batched_10x_ready_score(payload):
        raise ValueError("rust_batched_10x_ready_score mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start complete: or blocked:")
    if verdict != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal result artifact."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def upstream_gate_receipts(root: str | Path) -> JsonDict:
    """Validate upstream Exp5724 and Exp5738 artifacts."""

    root_path = Path(root)
    return {
        "exp5724": _upstream_receipt(
            root_path,
            exp5724.RESULT_RELATIVE_PATH,
            exp5724.validate_artifact,
            ready_field="rust_crossover_ready_score",
            ready_value=0.0,
            source_hashes=False,
        ),
        "exp5738": _upstream_receipt(
            root_path,
            exp5738.RESULT_RELATIVE_PATH,
            exp5738.validate_artifact,
            ready_field="batch_backend_ready_score",
            ready_value=1.0,
            source_hashes=True,
        ),
        "claim_scope": {
            "cpu_software_only": True,
            "gpu_speedup_claim_allowed": False,
            "hardware_speedup_claim_allowed": False,
            "fpga_or_tsu_allowed": False,
        },
    }


def software_receipt(root: str | Path) -> JsonDict:
    """Collect Python, Rust, extension, compiler, and source receipts."""

    root_path = Path(root)
    receipt = exp5738.software_receipt(root_path)
    receipt["source_hashes"]["experiment_5739"] = file_sha256(Path(__file__))
    return receipt


def build_profile(root: str | Path) -> JsonDict:
    """Return the frozen local build profile used for timing."""

    profile = exp5738.build_profile(Path(root))
    profile.update(
        {
            "benchmark_module_hash": file_sha256(Path(__file__)),
            "python_baseline_weakened": False,
            "end_to_end_overhead_subtracted": False,
            "gpu": False,
            "fpga_or_tsu": False,
        }
    )
    return profile


def cpu_topology(affinity: Mapping[str, Any]) -> JsonDict:
    """Return physical and logical CPU topology receipts."""

    return exp5738.cpu_topology({"current_cpus": affinity.get("initial_cpus", [])})


def cpu_affinity_receipt(*, freeze_affinity: bool) -> JsonDict:
    """Record one-core and fixed-core placements for both arms."""

    if not hasattr(os, "sched_getaffinity"):
        regimes = [
            {
                "regime_id": "one_physical_core",
                "cpus": [],
                "affinity_enforced": False,
                "blocked_reason": "sched_affinity_unavailable",
            },
            {
                "regime_id": "fixed_recorded_cores",
                "cpus": [],
                "affinity_enforced": False,
                "blocked_reason": "sched_affinity_unavailable",
            },
        ]
        return {
            "observable": False,
            "freeze_requested": freeze_affinity,
            "initial_cpus": [],
            "thread_regimes": regimes,
        }
    current = sorted(os.sched_getaffinity(0))
    one = current[:1]
    fixed = list(current)
    regimes = [
        {
            "regime_id": "one_physical_core",
            "cpus": one,
            "affinity_enforced": bool(freeze_affinity and one),
            "physical_core_policy": "first available logical CPU used as one physical-core placement",
        },
        {
            "regime_id": "fixed_recorded_cores",
            "cpus": fixed,
            "affinity_enforced": bool(freeze_affinity and fixed),
            "physical_core_policy": "all initially allowed CPUs recorded and reused for both arms",
        },
    ]
    return {
        "observable": True,
        "freeze_requested": freeze_affinity,
        "initial_cpus": current,
        "thread_regimes": regimes,
        "baseline_weakening_note": (
            "The one-core regime is preregistered. The fixed regime uses the initially "
            "allowed CPU set for both arms and adds no sleeps, throttles, or arm-specific limits."
        ),
    }


def thread_receipts(thread_regimes: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record matched thread policy without mutating the environment."""

    pools: list[JsonDict] = []
    try:
        from threadpoolctl import threadpool_info

        pools = [dict(row) for row in threadpool_info()]
    except Exception as exc:  # pragma: no cover - optional dependency/environment receipt.
        pools = [{"available": False, "blocked_reason": f"{type(exc).__name__}:{exc}"}]
    return {
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "mutated_environment": False,
        "same_process_same_thread_policy_for_arms": True,
        "thread_regimes": [
            {
                "regime_id": row["regime_id"],
                "cpus": list(row.get("cpus", [])),
                "affinity_enforced": bool(row.get("affinity_enforced")),
            }
            for row in thread_regimes
        ],
        "observed_threadpools": pools,
        "implementation_asymmetry_disclosed": (
            "Rust arm uses OneAxisRustBackend.sample_batch with prefer_rust=True; "
            "Python arm uses the exact fallback with prefer_rust=False. Both receive "
            "identical batch inputs, seed schedules, checkpoints, and validation."
        ),
    }


def preconditions_checked(
    root: Path,
    upstream: Mapping[str, Any],
    affinity: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[JsonDict]:
    """Record every gate checked before timing interpretation."""

    profile = build_profile(root)
    return [
        {
            "resource": "exp5724_terminal_null",
            "available": upstream.get("exp5724", {}).get("ready") is True,
            "details": upstream.get("exp5724", {}),
        },
        {
            "resource": "exp5738_batched_backend_ready",
            "available": upstream.get("exp5738", {}).get("ready") is True,
            "details": upstream.get("exp5738", {}),
        },
        {
            "resource": "exp5738_backend_build_hashes",
            "available": upstream.get("exp5738", {}).get("backend_build_hashes_match") is True,
            "details": upstream.get("exp5738", {}).get("source_hash_comparisons", []),
        },
        {
            "resource": "rust_extension",
            "available": profile.get("extension_present") is True,
            "details": profile,
        },
        {
            "resource": "sample_batch_api",
            "available": callable(getattr(OneAxisRustBackend(), "sample_batch", None)),
        },
        {
            "resource": "problem_sizes_48_96_192",
            "available": {48, 96, 192}.issubset(set(protocol.get("problem_sizes", []))),
            "details": protocol.get("problem_sizes", []),
        },
        {
            "resource": "thirty_independent_batches_per_cell",
            "available": int(protocol.get("measured_batch_count", 0)) >= 30,
            "details": protocol.get("measured_batch_count"),
        },
        {
            "resource": "cpu_affinity_regimes",
            "available": len(affinity.get("thread_regimes", [])) == 2,
            "details": affinity.get("thread_regimes", []),
        },
        {
            "resource": "no_artificial_baseline_weakening",
            "available": True,
            "details": "No sleeps, throttles, arm-specific thread limits, or overhead subtraction.",
        },
    ]


def arm_configs() -> JsonDict:
    """Return the two production arm configurations."""

    return {
        RUST_ARM: {
            "backend_class": "OneAxisRustBackend",
            "prefer_rust": True,
            "sampler_api": "OneAxisRustBackend.sample_batch",
            "expected_active_backend": ACTIVE_RUST_BACKEND,
            "algorithm": ONE_AXIS_ALGORITHM,
            "topology": ONE_AXIS_TOPOLOGY,
        },
        PYTHON_ARM: {
            "backend_class": "OneAxisRustBackend",
            "prefer_rust": False,
            "sampler_api": "OneAxisRustBackend.sample_batch",
            "expected_active_backend": ACTIVE_PYTHON_FALLBACK,
            "algorithm": ONE_AXIS_ALGORITHM,
            "topology": ONE_AXIS_TOPOLOGY,
            "baseline_weakened": False,
        },
    }


def batch_items_for(
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    batch_size: int,
    batch_index: int,
) -> list[JsonDict]:
    """Create identical independent batch inputs for both arms."""

    fields, couplings = exp5724.arrays_from_workload(workload)
    items: list[JsonDict] = []
    for item_index in range(int(batch_size)):
        seed = batch_item_seed(protocol, workload, int(batch_size), int(batch_index), item_index)
        items.append(
            {
                "workload_id": (
                    f"{workload['workload_id']}:batch{int(batch_index)}:item{item_index}"
                ),
                "biases": fields,
                "couplings": couplings,
                "n_samples": int(protocol["sample_sweeps"]),
                "config": descriptor_for_run(
                    seed=seed,
                    initial_states=exp5724.initial_states_for(workload, seed),
                    initial_labels=list(range(len(exp5714.BETA_LADDER))),
                    burn_in_sweeps=int(protocol["burn_in_sweeps"]),
                ),
            }
        )
    return items


def batch_item_seed(
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    batch_size: int,
    batch_index: int,
    item_index: int,
) -> int:
    """Derive a replayable item seed from the preregistered seed schedule."""

    base = int(protocol["random_seeds"][int(batch_index) % len(protocol["random_seeds"])])
    return _stable_seed64(
        "exp5739-item",
        base,
        workload["workload_id"],
        int(batch_size),
        int(batch_index),
        int(item_index),
    )


def cell_id(regime_id: str, workload: Mapping[str, Any], batch_size: int) -> str:
    """Return the stable cell identifier used across quality and timing rows."""

    return f"{regime_id}:{workload['workload_id']}:batch{int(batch_size)}"


def summarize_end_to_end(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize primary end-to-end latency by cell and arm."""

    return _summarize_record_values(records, value_key="end_to_end_s", output_key="samples_s")


def summarize_throughput(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize end-to-end throughput by cell and arm."""

    rows = _summarize_record_values(
        records,
        value_key="throughput_samples_per_s",
        output_key="throughput_samples_per_s",
    )
    for row in rows:
        matching = [
            record
            for record in records
            if record["cell_id"] == row["cell_id"] and record["arm"] == row["arm"]
        ]
        row["samples_per_batch"] = int(matching[0]["samples_per_batch"]) if matching else 0
        row["mean_samples_per_s"] = row["mean_s"]
        row.pop("mean_s", None)
    return rows


def summarize_phase_times(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize measured phases while declaring they remain in end-to-end timing."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        for phase, value in record["phase_s"].items():
            key = (str(record["cell_id"]), str(record["arm"]), str(phase))
            grouped[key].append(float(value))
            metadata[key] = record
    rows: list[JsonDict] = []
    for key, values in sorted(grouped.items()):
        first = metadata[key]
        rows.append(
            {
                "cell_id": key[0],
                "thread_regime": first["thread_regime"],
                "size": int(first["size"]),
                "family": first["family"],
                "batch_size": int(first["batch_size"]),
                "arm": key[1],
                "phase": key[2],
                "n": len(values),
                **seconds_summary(values),
                "samples_s": [_stable_float(value) for value in values],
                "included_in_end_to_end": True,
            }
        )
    return rows


def peak_rss_by_arm(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return peak process RSS by thread regime and arm."""

    result: JsonDict = {}
    regimes = sorted({str(row["thread_regime"]) for row in records})
    for regime in regimes:
        result[regime] = {}
        for arm in ARMS:
            values = [
                int(row["peak_rss_kib"])
                for row in records
                if row["thread_regime"] == regime and row["arm"] == arm
            ]
            result[regime][arm] = {
                "peak_kib": max(values) if values else None,
                "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            }
    return result


def seconds_summary(values: Sequence[float]) -> JsonDict:
    """Return stable summary statistics for timing or throughput samples."""

    vals = [float(value) for value in values]
    median = statistics.median(vals)
    mean = statistics.fmean(vals)
    deviations = [abs(value - median) for value in vals]
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean_s": _stable_float(mean),
        "median_s": _stable_float(median),
        "min_s": _stable_float(min(vals)),
        "max_s": _stable_float(max(vals)),
        "stdev_s": _stable_float(stdev),
        "mad_s": _stable_float(statistics.median(deviations)),
        "coefficient_of_variation": _stable_float(stdev / mean) if mean > 0 else 0.0,
    }


def excluded_pair_reasons(quality_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate all excluded quality rows by preregistered reason."""

    counts = Counter(
        str(row.get("excluded_reason"))
        for row in quality_rows
        if row.get("quality_matched") is not True
    )
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counts.items())
        if reason and reason != "None"
    ]


def current_peak_rss_kib() -> int:
    """Return process peak RSS in KiB on Linux-style hosts."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _summarize_record_values(
    records: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    output_key: str,
) -> list[JsonDict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["cell_id"]), str(record["arm"]))
        grouped[key].append(float(record[value_key]))
        metadata[key] = record
    rows: list[JsonDict] = []
    for key, values in sorted(grouped.items()):
        first = metadata[key]
        rows.append(
            {
                "cell_id": key[0],
                "thread_regime": first["thread_regime"],
                "size": int(first["size"]),
                "family": first["family"],
                "batch_size": int(first["batch_size"]),
                "arm": key[1],
                "n": len(values),
                **seconds_summary(values),
                output_key: [_stable_float(value) for value in values],
            }
        )
    return rows


def _items_from_record(record: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return list(record.get("items", []))


def _warm_cell(protocol: Mapping[str, Any], workload: Mapping[str, Any], batch_size: int) -> None:
    for warmup in range(int(protocol["warmup_count"])):
        items = batch_items_for(protocol, workload, batch_size, warmup)
        for prefer_rust in (True, False):
            OneAxisRustBackend(
                seed=int(items[0]["config"]["seed"]),
                prefer_rust=prefer_rust,
            ).sample_batch(items)


def _arm_order(regime_id: str, workload_id: str, batch_index: int) -> tuple[str, str]:
    seed = _stable_seed64("exp5739-arm-order", regime_id, workload_id, int(batch_index))
    return (RUST_ARM, PYTHON_ARM) if seed % 2 == 0 else (PYTHON_ARM, RUST_ARM)


@contextmanager
def _temporary_affinity(cpus: object, enabled: bool) -> Iterator[None]:
    if not enabled or not hasattr(os, "sched_getaffinity"):
        yield
        return
    previous = sorted(os.sched_getaffinity(0))
    selected = {int(cpu) for cpu in cpus} if isinstance(cpus, Sequence) else set()
    if not selected:
        yield
        return
    try:
        os.sched_setaffinity(0, selected)
        yield
    finally:
        os.sched_setaffinity(0, set(previous))


def _upstream_receipt(
    root: Path,
    relative_path: Path,
    validator: Callable[[Mapping[str, Any]], None],
    *,
    ready_field: str,
    ready_value: object,
    source_hashes: bool,
) -> JsonDict:
    path = root / relative_path
    receipt: JsonDict = {
        "path": relative_path.as_posix(),
        "available": path.exists(),
        "sha256": None,
        "valid": False,
        "ready": False,
        "ready_field": ready_field,
        "ready_value": None,
        "honest_verdict": None,
        "blocked_reason": None,
    }
    if not path.exists():
        receipt["blocked_reason"] = "missing_upstream_artifact"
        return receipt
    receipt["sha256"] = file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validator(payload)
    except Exception as exc:  # noqa: BLE001 - exact upstream failure is a receipt.
        receipt["blocked_reason"] = f"invalid_upstream:{type(exc).__name__}"
        return receipt
    value = payload.get(ready_field)
    receipt.update(
        {
            "valid": True,
            "ready_value": value,
            "honest_verdict": payload.get("honest_verdict"),
            "inference_substrate": payload.get("inference_substrate"),
        }
    )
    if source_hashes:
        comparisons = _source_hash_comparisons(root, payload)
        receipt["source_hash_comparisons"] = comparisons
        receipt["backend_build_hashes_match"] = all(
            row["matches"] for row in comparisons if row["gate"] == "backend_build"
        )
        receipt["driver_hash_matches"] = all(
            row["matches"] for row in comparisons if row["gate"] == "driver"
        )
        receipt["ready"] = value == ready_value and receipt["backend_build_hashes_match"] is True
    else:
        receipt["ready"] = value == ready_value
    return receipt


def _source_hash_comparisons(root: Path, payload: Mapping[str, Any]) -> list[JsonDict]:
    recorded = payload.get("software_receipt", {}).get("source_hashes", {})
    paths = {
        "experiment_5738": (
            root / "python/carnot/experiment_5738_one_axis_rust_batched_backend.py",
            "driver",
        ),
        "one_axis_backend": (
            root / "python/carnot/samplers/one_axis_rust_backend.py",
            "backend_build",
        ),
        "sampler_backend_factory": (
            root / "python/carnot/samplers/backend.py",
            "backend_build",
        ),
        "rust_one_axis_core": (
            root / "crates/carnot-samplers/src/one_axis_tempering.rs",
            "backend_build",
        ),
        "pyo3_one_axis_binding": (
            root / "crates/carnot-python/src/one_axis_tempering.rs",
            "backend_build",
        ),
    }
    rows: list[JsonDict] = []
    for key, (path, gate) in paths.items():
        recorded_hash = recorded.get(key)
        current = file_sha256(path) if path.exists() else None
        rows.append(
            {
                "source": key,
                "path": path.relative_to(root).as_posix(),
                "gate": gate if recorded_hash is not None else "not_recorded_in_exp5738",
                "recorded_sha256": recorded_hash,
                "current_sha256": current,
                "matches": bool(recorded_hash == current) if recorded_hash is not None else None,
            }
        )
    return rows


def _stable_seed64(*parts: object) -> int:
    digest = sha256_json([str(part) for part in parts])
    return int(digest[:16], 16) % (2**63)


def _stable_float(value: Any) -> float:
    return round(float(value), 12)


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
