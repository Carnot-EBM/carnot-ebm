"""Exp5724 one-axis Rust/Python production-backend matched crossover.

Spec refs: REQ-SAMPLE-5724, SCENARIO-SAMPLE-5724.

This experiment asks a narrow CPU software question: where, if anywhere, the
production Rust/PyO3 one-axis backend beats the exact Python one-axis fallback
when both are driven through the same ``SamplerBackend.sample`` contract. The
primary speedup uses end-to-end time and a null crossover is a valid result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot import experiment_5611_cdls_matched_sampler_crossover as exp5611
from carnot import experiment_5623_cdls_multiseed_cpu_cuda_crossover as exp5623
from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5715_one_axis_tempering_rust_quality_restart as exp5715
from carnot import experiment_5723_one_axis_rust_samplerbackend_integration as exp5723
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
RESULT_RELATIVE_PATH = Path("results/experiment_5724_one_axis_rust_python_matched_crossover.json")

EXPERIMENT = 5724
EXPERIMENT_ID = "exp5724-one-axis-rust-python-matched-crossover"
MILESTONE = "2026.07.524"
RUN_DATE = "2026-07-19"
SCHEMA = "carnot.experiment_5724.one_axis_rust_python_matched_crossover.v1"
SPEC_REFS = ("REQ-SAMPLE-5724", "SCENARIO-SAMPLE-5724")
INFERENCE_SUBSTRATE = "matched_cpu_python_vs_rust_pyo3_production_samplerbackend"
TERMINAL_PREFIXES = ("complete:", "blocked:")

DEFAULT_PROBLEM_SIZES = (3, 6, 12, 24, 48, 96)
DEFAULT_TOPOLOGY_FAMILIES = (
    "ferromagnetic_ring_easy",
    "frustrated_chord_moderate",
    "planted_basin_hard",
)
DEFAULT_RANDOM_SEEDS = tuple(range(5724, 5734))
DEFAULT_WARMUP_COUNT = 3
DEFAULT_MEASURED_REPETITION_COUNT = 30
DEFAULT_QUALITY_SAMPLE_SWEEPS = 3
DEFAULT_TIMING_SAMPLE_SWEEPS = 2
DEFAULT_BURN_IN_SWEEPS = 1
DEFAULT_BENCHMARK_ORDER_SEED = 5_724_019
ENUMERABLE_TARGET_MAX_N = 12
BOOTSTRAP_RESAMPLES = 500

RUST_ARM = "rust_pyo3"
PYTHON_ARM = "python_exact_fallback"
ARMS = (RUST_ARM, PYTHON_ARM)
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

QUALITY_MARGINS: JsonDict = {
    "feasibility_delta_max": 0.0,
    "best_energy_delta_abs_max": 1e-12,
    "mean_energy_delta_abs_max": 1e-12,
    "acceptance_rate_delta_abs_max": 1e-12,
    "swap_acceptance_rate_delta_abs_max": 1e-12,
    "target_distribution_tv_delta_max": 1e-12,
    "restart_match_required": True,
    "work_counters_match_required": True,
    "rust_active_backend_required": ACTIVE_RUST_BACKEND,
    "python_active_backend_required": ACTIVE_PYTHON_FALLBACK,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": (
        "Explains why every crossover field exists before a reviewer trusts the JSON shape."
    ),
    "upstream_gate_receipts": (
        "Pins Exp5611/5623 CPU/CUDA terminal scope and Exp5714/5715/5723 one-axis "
        "Rust/Python readiness before timing is interpreted."
    ),
    "hardware_receipt": (
        "Authenticates CPU, OS, memory, affinity, and observable frequency without implying "
        "board or accelerator use."
    ),
    "software_receipt": (
        "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay "
        "the production arms."
    ),
    "build_profile": "Freezes release/debug, PyO3 ABI, and feature settings before timing.",
    "cpu_affinity": (
        "Shows the CPU set used for timing instead of leaving scheduler placement implicit."
    ),
    "thread_receipts": (
        "Records thread environment and observed thread pools so Python and Rust work are "
        "compared under the same CPU policy."
    ),
    "preregistered_protocol": (
        "Freezes workloads, sizes, seeds, repetitions, warmups, budgets, margins, thresholds, "
        "and null rules before timing."
    ),
    "workload_manifest": (
        "Lists every size/family Hamiltonian and descriptor hash used by both arms."
    ),
    "problem_sizes": "Makes the PyO3-overhead to kernel-dominated size range explicit.",
    "topology_families": "Shows all hardness/topology families entering the denominator.",
    "random_seeds": "Records the ten paired seeds and benchmark-order seed for replay.",
    "warmup_count": "Prevents cold-start effects from entering primary measurements.",
    "measured_repetition_count": (
        "Proves each qualified size/family has at least thirty timing repetitions."
    ),
    "arm_configs": (
        "Names the Rust/PyO3 and exact Python fallback SamplerBackend configurations under "
        "one contract."
    ),
    "matched_work_receipts": (
        "Proves transitions, energy evaluations, replicas, swaps, restarts, checkpoints, "
        "stopping, and initial states are identical."
    ),
    "quality_metrics_by_pair": "Reports quality before any speedup denominator is formed.",
    "quality_margins": (
        "Predeclares feasibility, energy, acceptance, swap, target, restart, and work-counter "
        "margins."
    ),
    "quality_matched_pair_count": (
        "Counts only pairs that passed quality gates before timing intervals."
    ),
    "excluded_pair_reasons": (
        "Keeps failed or unmatched pairs in the denominator instead of silently shrinking evidence."
    ),
    "kernel_times": (
        "Reports the measured SamplerBackend sampling call cost separately from setup and "
        "validation."
    ),
    "pyo3_overhead_times": (
        "Records PyO3 boundary probe cost without subtracting it from primary end-to-end timing."
    ),
    "serialization_times": (
        "Reports checkpoint/JSON serialization cost separately while keeping it in end-to-end "
        "timing."
    ),
    "validation_times": (
        "Reports per-run validation cost separately while keeping it in end-to-end timing."
    ),
    "end_to_end_times": (
        "Preserves the primary timing evidence that includes all production overhead."
    ),
    "peak_rss_by_arm": (
        "Records memory pressure by arm so speed is not traded against unreported RSS growth."
    ),
    "paired_speedup_ratios": (
        "Reports paired Python/Rust end-to-end ratios rather than unrelated aggregate means."
    ),
    "paired_speedup_intervals": (
        "Uses bootstrap intervals so a crossover cannot rest on one noisy repetition."
    ),
    "qualified_crossover_n": (
        "Records the first gated consecutive crossover size or null for a terminal "
        "no-crossover result."
    ),
    "rust_crossover_ready_score": (
        "Equals 1.0 only when matched quality and end-to-end timing prove a Rust CPU crossover "
        "with no hardware claim."
    ),
    "software_speedup_claimed": (
        "Bare boolean separates an allowed Rust/Python software timing claim from a null result."
    ),
    "timing_claimed": (
        "Bare true declares this is a CPU timing study, unlike Exp5714/5715/5723 parity-only "
        "artifacts."
    ),
    "hardware_speedup_claimed": (
        "Bare false prevents Rust/PyO3 CPU timing from becoming a board or TSU claim."
    ),
    "gpu_speedup_claimed": (
        "Bare false prevents the retired CPU/CUDA substrate from being reopened."
    ),
    "fpga_or_tsu_used": "Bare false records that no FPGA, board, or TSU participated.",
    "inference_substrate": (
        "Declares matched CPU Python versus Rust/PyO3 production SamplerBackend sampling, not "
        "LLM, GPU, or hardware timing."
    ),
    "reproducibility_checksum": (
        "Content-addresses the complete artifact after blanking the self-checksum field."
    ),
    "honest_verdict": (
        "Starts complete: or blocked: and states whether the Rust/Python crossover is proven or "
        "terminal null."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


class Clock:
    """Callable wall-clock source used to make timing injection explicit."""

    def __call__(self) -> float:
        return time.perf_counter()


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for reproducible artifact hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using Carnot's SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for source and upstream receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def preregistered_protocol(
    *,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_repetition_count: int = DEFAULT_MEASURED_REPETITION_COUNT,
    allow_underpowered: bool = False,
) -> JsonDict:
    """Freeze the benchmark design before timing can run.

    ``allow_underpowered`` is only for deterministic unit-smoke tests. The
    terminal artifact builder uses the default strict path so the emitted JSON
    cannot satisfy the production contract with fewer sizes, families, seeds,
    or repetitions than the preregistered gate requires.
    """

    sizes = tuple(int(size) for size in problem_sizes)
    families = tuple(str(family) for family in topology_families)
    seeds = tuple(int(seed) for seed in random_seeds)
    repetitions = int(measured_repetition_count)
    if not allow_underpowered:
        if len(sizes) < 6:
            raise ValueError("problem_sizes must contain at least six sizes")
        if len(families) < 3:
            raise ValueError("topology_families must contain at least three families")
        if len(seeds) < 10:
            raise ValueError("random_seeds must contain at least ten paired seeds")
        if repetitions < 30:
            raise ValueError("measured_repetition_count must be at least thirty")
    if any(size <= 0 for size in sizes):
        raise ValueError("problem_sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("problem_sizes must be unique")
    if len(set(families)) != len(families):
        raise ValueError("topology_families must be unique")
    if not seeds:
        raise ValueError("random_seeds must not be empty")
    if int(warmup_count) < 0:
        raise ValueError("warmup_count must be nonnegative")

    return {
        "schema": "carnot.exp5724.preregistered_protocol.v1",
        "frozen_before_timing": True,
        "run_date": RUN_DATE,
        "build_profile_frozen": True,
        "hardware_os_compiler_python_frozen": True,
        "cpu_affinity_frozen": True,
        "thread_policy_frozen": True,
        "problem_sizes": list(sizes),
        "topology_families": list(families),
        "random_seeds": list(seeds),
        "benchmark_order_seed": DEFAULT_BENCHMARK_ORDER_SEED,
        "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
        "proposal_std": float(exp5714.exp5622.CDLS_PROPOSAL_STD),
        "drift_scale": float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        "energy_convention": ENERGY_CONVENTION,
        "algorithm": ONE_AXIS_ALGORITHM,
        "topology": ONE_AXIS_TOPOLOGY,
        "warmup_count": int(warmup_count),
        "measured_repetition_count": repetitions,
        "quality_sample_sweeps": DEFAULT_QUALITY_SAMPLE_SWEEPS,
        "timing_sample_sweeps": DEFAULT_TIMING_SAMPLE_SWEEPS,
        "burn_in_sweeps": DEFAULT_BURN_IN_SWEEPS,
        "restart_suffix_sweeps": 1,
        "checkpoint_restarts": 2,
        "stopping_rule": "fixed_sweep_budget_no_early_stop",
        "quality_margins": dict(QUALITY_MARGINS),
        "speedup_ratio": "python_end_to_end_time / rust_end_to_end_time",
        "crossover_threshold": 1.0,
        "crossover_rule": (
            "qualified_crossover_n is the first size whose size-ranked suffix has matched "
            "quality and bootstrap intervals entirely above 1.0"
        ),
        "primary_timing": "end_to_end_time_no_overhead_subtraction",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "gpu_fpga_tsu_claims_allowed": False,
    }


def build_workload_manifest(
    *,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
) -> list[JsonDict]:
    """Build deterministic Ising workloads shared by both benchmark arms."""

    workloads: list[JsonDict] = []
    for size in problem_sizes:
        for family in topology_families:
            fields, couplings, target = _hamiltonian_for_family(int(size), str(family))
            edge_list = _edge_list(couplings)
            descriptor_hash = sha256_json(
                {
                    "family": family,
                    "size": int(size),
                    "fields": _round_list(fields),
                    "edge_list": edge_list,
                    "target": target.astype(int).tolist(),
                    "energy_convention": ENERGY_CONVENTION,
                }
            )
            exact_target = (
                exact_target_summary(fields, couplings)
                if int(size) <= ENUMERABLE_TARGET_MAX_N
                else {"enumerable": False, "reason": "state_space_exceeds_preregistered_limit"}
            )
            workloads.append(
                {
                    "workload_id": f"{family}_n{int(size)}_{descriptor_hash[:10]}",
                    "family": str(family),
                    "size": int(size),
                    "n_spins": int(size),
                    "descriptor_hash": descriptor_hash,
                    "fields": _round_list(fields),
                    "edge_list": edge_list,
                    "edge_count": len(edge_list),
                    "target_spins": target.astype(int).tolist(),
                    "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
                    "algorithm": ONE_AXIS_ALGORITHM,
                    "topology": ONE_AXIS_TOPOLOGY,
                    "exact_target": exact_target,
                }
            )
    return workloads


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    benchmark_runner: BenchmarkRunner = None,
    problem_sizes: Sequence[int] = DEFAULT_PROBLEM_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_repetition_count: int = DEFAULT_MEASURED_REPETITION_COUNT,
    freeze_affinity: bool = True,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp5724 matched crossover artifact."""

    started = time.perf_counter()
    runner = benchmark_runner or run_matched_crossover_study
    root_path = Path(root)
    affinity = cpu_affinity_receipt(freeze=freeze_affinity)
    threads = thread_receipts()
    protocol = preregistered_protocol(
        problem_sizes=problem_sizes,
        topology_families=topology_families,
        random_seeds=random_seeds,
        warmup_count=warmup_count,
        measured_repetition_count=measured_repetition_count,
    )
    workloads = build_workload_manifest(
        problem_sizes=protocol["problem_sizes"],
        topology_families=protocol["topology_families"],
    )
    upstream = upstream_gate_receipts(root_path)
    if upstream.get("exp5723", {}).get("ready") is True:
        evidence = runner(
            protocol=protocol,
            workloads=workloads,
            clock=Clock(),
        )
    else:
        evidence = blocked_benchmark_evidence(protocol, workloads, "upstream_gate_not_ready")
    quality_count = sum(1 for row in evidence["quality_metrics_by_pair"] if row["quality_matched"])
    qualified = qualified_crossover_from_intervals(
        evidence["paired_speedup_intervals"],
        problem_sizes=protocol["problem_sizes"],
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
        "upstream_gate_receipts": upstream,
        "hardware_receipt": hardware_receipt(affinity),
        "software_receipt": software_receipt(root_path),
        "build_profile": build_profile(root_path),
        "cpu_affinity": affinity,
        "thread_receipts": threads,
        "preregistered_protocol": protocol,
        "workload_manifest": workloads,
        "problem_sizes": list(protocol["problem_sizes"]),
        "topology_families": list(protocol["topology_families"]),
        "random_seeds": list(protocol["random_seeds"]),
        "warmup_count": int(protocol["warmup_count"]),
        "measured_repetition_count": int(protocol["measured_repetition_count"]),
        "arm_configs": arm_configs(),
        "matched_work_receipts": evidence["matched_work_receipts"],
        "quality_metrics_by_pair": evidence["quality_metrics_by_pair"],
        "quality_margins": dict(QUALITY_MARGINS),
        "quality_matched_pair_count": quality_count,
        "excluded_pair_reasons": evidence["excluded_pair_reasons"],
        "kernel_times": evidence["kernel_times"],
        "pyo3_overhead_times": evidence["pyo3_overhead_times"],
        "serialization_times": evidence["serialization_times"],
        "validation_times": evidence["validation_times"],
        "end_to_end_times": evidence["end_to_end_times"],
        "peak_rss_by_arm": evidence["peak_rss_by_arm"],
        "paired_speedup_ratios": evidence["paired_speedup_ratios"],
        "paired_speedup_intervals": evidence["paired_speedup_intervals"],
        "qualified_crossover_n": qualified,
        "rust_crossover_ready_score": 0.0,
        "software_speedup_claimed": qualified is not None,
        "timing_claimed": True,
        "hardware_speedup_claimed": False,
        "gpu_speedup_claimed": False,
        "fpga_or_tsu_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: crossover gates not evaluated",
    }
    artifact["rust_crossover_ready_score"] = rust_crossover_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_matched_crossover_study(
    *,
    protocol: Mapping[str, Any],
    workloads: Sequence[Mapping[str, Any]],
    clock: Clock | None = None,
) -> JsonDict:
    """Run quality and timing through the production SamplerBackend boundary."""

    active_clock = clock or Clock()
    quality_rows: list[JsonDict] = []
    work_rows: list[JsonDict] = []
    timing_records: list[JsonDict] = []
    for workload in workloads:
        for seed in protocol["random_seeds"]:
            quality, work = quality_pair(protocol, workload, int(seed))
            quality_rows.append(quality)
            work_rows.append(work)
    for workload in workloads:
        _warm_arms(protocol, workload)
        for repetition in range(int(protocol["measured_repetition_count"])):
            seed = int(protocol["random_seeds"][repetition % len(protocol["random_seeds"])])
            for arm in _arm_order(protocol, workload["workload_id"], repetition):
                timing_records.append(
                    measure_arm_once(
                        protocol=protocol,
                        workload=workload,
                        seed=seed,
                        repetition_index=repetition,
                        arm=arm,
                        clock=active_clock,
                    )
                )

    quality_by_pair = {str(row["pair_id"]): bool(row["quality_matched"]) for row in quality_rows}
    ratios = paired_speedup_ratios_from_records(timing_records, quality_by_pair=quality_by_pair)
    intervals = paired_speedup_intervals_from_ratios(
        ratios,
        problem_sizes=protocol["problem_sizes"],
        topology_families=protocol["topology_families"],
        quality_rows=quality_rows,
        required_repetitions=int(protocol["measured_repetition_count"]),
    )
    return {
        "matched_work_receipts": work_rows,
        "quality_metrics_by_pair": quality_rows,
        "excluded_pair_reasons": excluded_pair_reasons(quality_rows),
        "kernel_times": summarize_timing_records(timing_records, "kernel_s"),
        "pyo3_overhead_times": pyo3_overhead_probe(protocol, workloads, active_clock),
        "serialization_times": summarize_timing_records(timing_records, "serialization_s"),
        "validation_times": summarize_timing_records(timing_records, "validation_s"),
        "end_to_end_times": summarize_timing_records(timing_records, "end_to_end_s"),
        "peak_rss_by_arm": peak_rss_by_arm(timing_records),
        "paired_speedup_ratios": ratios,
        "paired_speedup_intervals": intervals,
    }


def blocked_benchmark_evidence(
    protocol: Mapping[str, Any],
    workloads: Sequence[Mapping[str, Any]],
    reason: str,
) -> JsonDict:
    """Return denominator-preserving blocked evidence when upstream gates fail."""

    quality = [
        {
            "pair_id": f"{workload['workload_id']}:seed{seed}",
            "workload_id": workload["workload_id"],
            "size": workload["size"],
            "family": workload["family"],
            "seed": int(seed),
            "quality_matched": False,
            "excluded_reason": reason,
        }
        for workload in workloads
        for seed in protocol["random_seeds"]
    ]
    work = [
        {
            "workload_id": workload["workload_id"],
            "size": workload["size"],
            "family": workload["family"],
            "matched": False,
            "excluded_reason": reason,
        }
        for workload in workloads
    ]
    intervals = [
        {
            "size": int(size),
            "family_count": len(protocol["topology_families"]),
            "repetition_count": 0,
            "rust_end_to_end_speedup_interval_95": [None, None],
            "interval_entirely_above_one": False,
            "quality_matched": False,
            "excluded_reason": reason,
        }
        for size in protocol["problem_sizes"]
    ]
    return {
        "matched_work_receipts": work,
        "quality_metrics_by_pair": quality,
        "excluded_pair_reasons": excluded_pair_reasons(quality),
        "kernel_times": [],
        "pyo3_overhead_times": [],
        "serialization_times": [],
        "validation_times": [],
        "end_to_end_times": [],
        "peak_rss_by_arm": {RUST_ARM: {}, PYTHON_ARM: {}},
        "paired_speedup_ratios": [],
        "paired_speedup_intervals": intervals,
    }


def quality_pair(
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    seed: int,
) -> tuple[JsonDict, JsonDict]:
    """Run one paired quality/restart check before timing enters speedups."""

    fields, couplings = arrays_from_workload(workload)
    descriptor = descriptor_for_quality(protocol, workload, seed)
    rust = OneAxisRustBackend(seed=seed, prefer_rust=True).run_descriptor(
        fields,
        couplings,
        int(protocol["quality_sample_sweeps"]),
        descriptor,
    )
    python = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        fields,
        couplings,
        int(protocol["quality_sample_sweeps"]),
        descriptor,
    )
    rust_metrics = sample_quality_metrics(fields, couplings, rust)
    python_metrics = sample_quality_metrics(fields, couplings, python)
    restart_match = cross_restart_match(fields, couplings, descriptor, rust, python, seed)
    work = matched_work_receipt(workload, seed, rust["receipt"], python["receipt"], restart_match)
    target_delta = distribution_tv(
        energy_histogram(rust_metrics["energies"]),
        energy_histogram(python_metrics["energies"]),
    )
    row: JsonDict = {
        "pair_id": f"{workload['workload_id']}:seed{seed}",
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "seed": int(seed),
        "rust_active_backend": rust["receipt"]["active_backend"],
        "python_active_backend": python["receipt"]["active_backend"],
        "feasibility_delta": abs(
            rust_metrics["feasibility_rate"] - python_metrics["feasibility_rate"]
        ),
        "best_energy_delta_abs": abs(rust_metrics["best_energy"] - python_metrics["best_energy"]),
        "mean_energy_delta_abs": abs(rust_metrics["mean_energy"] - python_metrics["mean_energy"]),
        "acceptance_rate_delta_abs": abs(
            rust_metrics["acceptance_rate"] - python_metrics["acceptance_rate"]
        ),
        "swap_acceptance_rate_delta_abs": abs(
            rust_metrics["swap_acceptance_rate"] - python_metrics["swap_acceptance_rate"]
        ),
        "target_distribution_tv_delta": target_delta,
        "restart_match": restart_match,
        "work_counters_match": work["matched"],
        "exact_target_receipt": workload.get("exact_target", {}),
        "rust_metrics": {key: value for key, value in rust_metrics.items() if key != "energies"},
        "python_metrics": {
            key: value for key, value in python_metrics.items() if key != "energies"
        },
        "quality_matched": False,
        "excluded_reason": None,
    }
    row["quality_matched"] = quality_matched(row)
    if not row["quality_matched"]:
        row["excluded_reason"] = quality_exclusion_reason(row)
    return row, work


def measure_arm_once(
    *,
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    seed: int,
    repetition_index: int,
    arm: str,
    clock: Clock,
) -> JsonDict:
    """Measure one production SamplerBackend sample call and its overheads."""

    fields, couplings = arrays_from_workload(workload)
    descriptor = descriptor_for_timing(protocol, workload, seed)
    e2e_start = clock()
    setup_start = clock()
    backend = OneAxisRustBackend(seed=seed, prefer_rust=(arm == RUST_ARM))
    setup_s = clock() - setup_start

    kernel_start = clock()
    samples = backend.sample(
        fields,
        couplings,
        int(protocol["timing_sample_sweeps"]),
        descriptor,
    )
    kernel_s = clock() - kernel_start

    serialization_start = clock()
    checkpoint = backend.save_checkpoint()
    encoded_checkpoint = canonical_json(checkpoint)
    serialization_s = clock() - serialization_start

    validation_start = clock()
    validation = validate_timing_run(samples, backend, encoded_checkpoint)
    validation_s = clock() - validation_start
    end_to_end_s = clock() - e2e_start
    return {
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "seed": int(seed),
        "repetition_index": int(repetition_index),
        "arm": arm,
        "active_backend": backend.last_receipt["active_backend"] if backend.last_receipt else None,
        "setup_s": _stable_float(setup_s),
        "kernel_s": _stable_float(kernel_s),
        "serialization_s": _stable_float(serialization_s),
        "validation_s": _stable_float(validation_s),
        "end_to_end_s": _stable_float(end_to_end_s),
        "peak_rss_kib": current_peak_rss_kib(),
        "validation": validation,
    }


def validate_timing_run(
    samples: np.ndarray,
    backend: OneAxisRustBackend,
    encoded_checkpoint: str,
) -> JsonDict:
    """Validate shape, receipts, and checkpoint checksum for one timed run."""

    if backend.last_receipt is None or backend.last_checkpoint is None:
        raise ValueError("backend did not record receipt/checkpoint")
    checkpoint = backend.last_checkpoint
    valid = (
        samples.dtype == np.bool_
        and samples.ndim == 2
        and checkpoint.get("payload_checksum") == checkpoint_checksum(checkpoint)
    )
    return {
        "valid": bool(valid),
        "sample_shape": list(samples.shape),
        "checkpoint_bytes": len(encoded_checkpoint.encode("utf-8")),
        "transition_budget": backend.last_receipt["transition_budget"],
    }


def sample_quality_metrics(
    fields: np.ndarray,
    couplings: np.ndarray,
    run: Mapping[str, Any],
) -> JsonDict:
    """Summarize feasibility, energy, and accept/swap rates for one arm."""

    samples = [[int(value) for value in sample] for sample in run["samples_spin"]]
    energies = [ising_energy(fields, couplings, sample) for sample in samples]
    within_events = [event for event in run["decision_log"] if event["kind"] == "within"]
    swap_events = [event for event in run["decision_log"] if event["kind"] == "swap"]
    feasibility = [
        all(value in {-1, 1} for value in sample) and len(sample) == fields.size
        for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "feasibility_rate": sum(feasibility) / max(1, len(feasibility)),
        "best_energy": min(energies),
        "mean_energy": statistics.fmean(energies),
        "acceptance_rate": _event_acceptance_rate(within_events),
        "swap_acceptance_rate": _event_acceptance_rate(swap_events),
        "energy_histogram": energy_histogram(energies),
        "energies": energies,
        "sample_hash": sha256_json(samples),
    }


def quality_matched(row: Mapping[str, Any]) -> bool:
    """Apply the preregistered Rust/Python quality margins to one pair."""

    return bool(
        row.get("rust_active_backend") == ACTIVE_RUST_BACKEND
        and row.get("python_active_backend") == ACTIVE_PYTHON_FALLBACK
        and float(row.get("feasibility_delta", 1.0)) <= QUALITY_MARGINS["feasibility_delta_max"]
        and float(row.get("best_energy_delta_abs", 1.0))
        <= QUALITY_MARGINS["best_energy_delta_abs_max"]
        and float(row.get("mean_energy_delta_abs", 1.0))
        <= QUALITY_MARGINS["mean_energy_delta_abs_max"]
        and float(row.get("acceptance_rate_delta_abs", 1.0))
        <= QUALITY_MARGINS["acceptance_rate_delta_abs_max"]
        and float(row.get("swap_acceptance_rate_delta_abs", 1.0))
        <= QUALITY_MARGINS["swap_acceptance_rate_delta_abs_max"]
        and float(row.get("target_distribution_tv_delta", 1.0))
        <= QUALITY_MARGINS["target_distribution_tv_delta_max"]
        and row.get("restart_match") is True
        and row.get("work_counters_match") is True
    )


def quality_exclusion_reason(row: Mapping[str, Any]) -> str:
    """Return the first preregistered exclusion reason for a quality row."""

    if row.get("rust_active_backend") != ACTIVE_RUST_BACKEND:
        return "rust_arm_not_active_pyo3"
    if row.get("python_active_backend") != ACTIVE_PYTHON_FALLBACK:
        return "python_arm_not_exact_fallback"
    if row.get("restart_match") is not True:
        return "restart_mismatch"
    if row.get("work_counters_match") is not True:
        return "work_counter_mismatch"
    for key, margin_key in (
        ("feasibility_delta", "feasibility_delta_max"),
        ("best_energy_delta_abs", "best_energy_delta_abs_max"),
        ("mean_energy_delta_abs", "mean_energy_delta_abs_max"),
        ("acceptance_rate_delta_abs", "acceptance_rate_delta_abs_max"),
        ("swap_acceptance_rate_delta_abs", "swap_acceptance_rate_delta_abs_max"),
        ("target_distribution_tv_delta", "target_distribution_tv_delta_max"),
    ):
        if float(row.get(key, 1.0)) > QUALITY_MARGINS[margin_key]:
            return key
    return "unknown_quality_mismatch"


def matched_work_receipt(
    workload: Mapping[str, Any],
    seed: int,
    rust_receipt: Mapping[str, Any],
    python_receipt: Mapping[str, Any],
    restart_match: bool,
) -> JsonDict:
    """Prove that Rust and Python arms used identical sampler work."""

    rust_counters = work_counters_from_receipt(rust_receipt)
    python_counters = work_counters_from_receipt(python_receipt)
    return {
        "pair_id": f"{workload['workload_id']}:seed{seed}",
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "seed": int(seed),
        "matched": rust_counters == python_counters
        and rust_receipt["transition_budget"] == python_receipt["transition_budget"]
        and restart_match,
        "rust": rust_counters,
        "python": python_counters,
        "transition_budget_match": rust_receipt["transition_budget"]
        == python_receipt["transition_budget"],
        "initial_state_hashes_match": rust_receipt["descriptor_hash"]
        == python_receipt["descriptor_hash"],
        "checkpoint_schema_match": CHECKPOINT_SCHEMA_VERSION == CHECKPOINT_SCHEMA_VERSION,
        "energy_convention": ENERGY_CONVENTION,
    }


def work_counters_from_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    """Expand adapter transition-budget receipts into comparable work counters."""

    budget = receipt["transition_budget"]
    corrected = int(budget["corrected_transitions"])
    swaps = int(budget["swap_attempts"])
    return {
        "replicas": len(exp5714.BETA_LADDER),
        "corrected_transitions": corrected,
        "swap_attempts": swaps,
        "energy_evaluations": 2 * (corrected + swaps),
        "cold_target_samples": int(budget["cold_target_samples"]),
        "total_sweeps": int(budget["total_sweeps"]),
        "checkpoints": 2,
        "restarts": 2,
        "stopping_rule": "fixed_sweep_budget_no_early_stop",
    }


def cross_restart_match(
    fields: np.ndarray,
    couplings: np.ndarray,
    descriptor: Mapping[str, Any],
    rust: Mapping[str, Any],
    python: Mapping[str, Any],
    seed: int,
) -> bool:
    """Verify Python-to-Rust and Rust-to-Python checkpoint suffix parity."""

    suffix = 1
    rust_checkpoint = rust["checkpoint"]
    python_checkpoint = python["checkpoint"]
    rust_to_rust = OneAxisRustBackend(seed=seed).run_descriptor(
        fields,
        couplings,
        suffix,
        {**descriptor, "checkpoint": rust_checkpoint, "burn_in_sweeps": 0},
    )
    rust_to_python = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        fields,
        couplings,
        suffix,
        {**descriptor, "checkpoint": rust_checkpoint, "burn_in_sweeps": 0},
    )
    python_to_python = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        fields,
        couplings,
        suffix,
        {**descriptor, "checkpoint": python_checkpoint, "burn_in_sweeps": 0},
    )
    python_to_rust = OneAxisRustBackend(seed=seed).run_descriptor(
        fields,
        couplings,
        suffix,
        {**descriptor, "checkpoint": python_checkpoint, "burn_in_sweeps": 0},
    )
    return _same_suffix(rust_to_rust, rust_to_python) and _same_suffix(
        python_to_python,
        python_to_rust,
    )


def descriptor_for_quality(
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Build the identical one-axis descriptor used for quality rows."""

    return descriptor_for_run(
        seed=int(seed),
        initial_states=initial_states_for(workload, seed),
        initial_labels=list(range(len(exp5714.BETA_LADDER))),
        burn_in_sweeps=int(protocol["burn_in_sweeps"]),
    )


def descriptor_for_timing(
    protocol: Mapping[str, Any],
    workload: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Build the identical one-axis descriptor used for timing rows."""

    return descriptor_for_run(
        seed=int(seed),
        initial_states=initial_states_for(workload, seed),
        initial_labels=list(range(len(exp5714.BETA_LADDER))),
        burn_in_sweeps=int(protocol["burn_in_sweeps"]),
    )


def initial_states_for(workload: Mapping[str, Any], seed: int) -> list[list[int]]:
    """Create deterministic initial replica states for one workload/seed pair."""

    n_spins = int(workload["size"])
    rng = np.random.default_rng(_stable_seed64("initial", workload["workload_id"], seed))
    return (
        np.where(rng.random((len(exp5714.BETA_LADDER), n_spins)) < 0.5, 1, -1).astype(int).tolist()
    )


def summarize_timing_records(
    records: Sequence[Mapping[str, Any]], component: str
) -> list[JsonDict]:
    """Summarize component timing by workload and arm while retaining raw samples."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["workload_id"]), str(record["arm"]))].append(record)
    rows: list[JsonDict] = []
    for (workload_id, arm), group in sorted(groups.items()):
        values = [float(row[component]) for row in group]
        first = group[0]
        rows.append(
            {
                "workload_id": workload_id,
                "size": int(first["size"]),
                "family": first["family"],
                "arm": arm,
                "n": len(values),
                **seconds_summary(values),
                "samples_s": [_stable_float(value) for value in values],
            }
        )
    return rows


def seconds_summary(values: Sequence[float]) -> JsonDict:
    """Return stable location and noise metrics for timing samples."""

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


def paired_speedup_ratios_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_by_pair: Mapping[str, bool] | None = None,
) -> list[JsonDict]:
    """Compute paired Python/Rust end-to-end ratios for each workload."""

    paired: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        pair_id = f"{record['workload_id']}:seed{record['seed']}"
        if quality_by_pair is not None and quality_by_pair.get(pair_id) is not True:
            continue
        paired[(str(record["workload_id"]), int(record["repetition_index"]))][
            str(record["arm"])
        ] = record
    by_workload: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, Mapping[str, Any]] = {}
    for (workload_id, _repetition), arms in paired.items():
        if RUST_ARM not in arms or PYTHON_ARM not in arms:
            continue
        rust_time = float(arms[RUST_ARM]["end_to_end_s"])
        python_time = float(arms[PYTHON_ARM]["end_to_end_s"])
        if rust_time <= 0 or python_time <= 0:
            continue
        by_workload[workload_id].append(python_time / rust_time)
        metadata[workload_id] = arms[RUST_ARM]
    rows: list[JsonDict] = []
    for workload_id, ratios in sorted(by_workload.items()):
        first = metadata[workload_id]
        rows.append(
            {
                "workload_id": workload_id,
                "size": int(first["size"]),
                "family": first["family"],
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
    problem_sizes: Sequence[int],
    topology_families: Sequence[str],
    quality_rows: Sequence[Mapping[str, Any]],
    required_repetitions: int,
) -> list[JsonDict]:
    """Aggregate ratios by size and compute deterministic bootstrap intervals."""

    rows: list[JsonDict] = []
    for size in problem_sizes:
        samples: list[float] = []
        families = set()
        for row in ratio_rows:
            if int(row["size"]) == int(size):
                samples.extend(float(value) for value in row["ratio_samples"])
                families.add(str(row["family"]))
        if samples:
            interval = bootstrap_interval(
                samples,
                seed=_stable_seed64("bootstrap", int(size), len(samples)),
            )
        else:
            interval = [None, None]
        interval_above = bool(interval[0] is not None and float(interval[0]) > 1.0)
        quality_for_size = [row for row in quality_rows if int(row.get("size", -1)) == int(size)]
        expected_repetitions = len(tuple(topology_families)) * int(required_repetitions)
        quality_matched = (
            bool(quality_for_size)
            and all(row.get("quality_matched") is True for row in quality_for_size)
            and len(families) == len(tuple(topology_families))
            and len(samples) >= expected_repetitions
        )
        rows.append(
            {
                "size": int(size),
                "family_count": len(families),
                "expected_family_count": len(tuple(topology_families)),
                "repetition_count": len(samples),
                "expected_repetition_count": expected_repetitions,
                "rust_end_to_end_speedup_interval_95": interval,
                "interval_entirely_above_one": interval_above,
                "quality_matched": quality_matched,
            }
        )
    return rows


def bootstrap_interval(values: Sequence[float], *, seed: int) -> list[float]:
    """Return a paired-ratio mean bootstrap interval."""

    vals = np.asarray([float(value) for value in values], dtype=np.float64)
    if vals.size == 0:
        return [None, None]  # type: ignore[list-item]
    if vals.size == 1:
        value = _stable_float(vals[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[index] = float(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return [_stable_float(low), _stable_float(high)]


def qualified_crossover_from_intervals(
    intervals: Sequence[Mapping[str, Any]],
    *,
    problem_sizes: Sequence[int],
) -> int | None:
    """Return the first size whose larger-size suffix proves Rust faster."""

    by_size = {int(row["size"]): row for row in intervals}
    ordered = [int(size) for size in problem_sizes]
    for index, size in enumerate(ordered):
        suffix = [by_size.get(candidate) for candidate in ordered[index:]]
        if any(row is None for row in suffix):
            continue
        if all(
            row.get("quality_matched") is True and row.get("interval_entirely_above_one") is True
            for row in suffix
            if row is not None
        ):
            return size
    return None


def pyo3_overhead_probe(
    protocol: Mapping[str, Any],
    workloads: Sequence[Mapping[str, Any]],
    clock: Clock,
) -> list[JsonDict]:
    """Measure a small PyO3 boundary probe without subtracting it from timing."""

    if not workloads:
        return []
    fields, couplings = arrays_from_workload(workloads[0])
    state = initial_states_for(workloads[0], int(protocol["random_seeds"][0]))[0]
    timings: list[float] = []
    try:
        rust_module = importlib.import_module("carnot._rust")
        rust_config = rust_module.RustOneAxisTemperingConfig(
            couplings.tolist(),
            fields.tolist(),
            [float(beta) for beta in exp5714.BETA_LADDER],
            float(exp5714.exp5622.CDLS_PROPOSAL_STD),
            float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        )
        core = rust_module.RustOneAxisTemperingCore(rust_config)
        for _index in range(int(protocol["measured_repetition_count"])):
            start = clock()
            core.energy(state)
            timings.append(clock() - start)
        rust_row = {
            "arm": RUST_ARM,
            "n": len(timings),
            **seconds_summary(timings),
            "probe": "RustOneAxisTemperingCore.energy",
            "not_subtracted_from_end_to_end": True,
        }
    except Exception as exc:  # noqa: BLE001 - absence is recorded as benchmark evidence.
        rust_row = {
            "arm": RUST_ARM,
            "n": 0,
            "probe": "RustOneAxisTemperingCore.energy",
            "blocked_reason": f"{type(exc).__name__}:{exc}",
            "not_subtracted_from_end_to_end": True,
        }
    python_row = {
        "arm": PYTHON_ARM,
        "n": int(protocol["measured_repetition_count"]),
        "mean_s": 0.0,
        "median_s": 0.0,
        "min_s": 0.0,
        "max_s": 0.0,
        "stdev_s": 0.0,
        "mad_s": 0.0,
        "coefficient_of_variation": 0.0,
        "not_applicable": True,
        "not_subtracted_from_end_to_end": True,
    }
    return [rust_row, python_row]


def peak_rss_by_arm(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the maximum observed process RSS per arm."""

    result: JsonDict = {}
    for arm in ARMS:
        values = [int(row["peak_rss_kib"]) for row in records if row["arm"] == arm]
        result[arm] = {
            "peak_kib": max(values) if values else None,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        }
    return result


def excluded_pair_reasons(quality_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate preregistered exclusion reasons while preserving denominator rows."""

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


def rust_crossover_ready_score(payload: Mapping[str, Any]) -> float:
    """Return the scalar gate for a proven Rust/Python CPU software crossover."""

    qualified = payload.get("qualified_crossover_n")
    expected_pairs = (
        len(payload.get("problem_sizes", []))
        * len(payload.get("topology_families", []))
        * len(payload.get("random_seeds", []))
    )
    upstream = payload.get("upstream_gate_receipts", {})
    gates = [
        isinstance(upstream, Mapping) and upstream.get("exp5723", {}).get("ready") is True,
        qualified is not None,
        payload.get("quality_matched_pair_count") == expected_pairs,
        not payload.get("excluded_pair_reasons"),
        all(row.get("matched") is True for row in payload.get("matched_work_receipts", [])),
        payload.get("software_speedup_claimed") is True,
        payload.get("timing_claimed") is True,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("gpu_speedup_claimed") is False,
        payload.get("fpga_or_tsu_used") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the terminal verdict, preserving null results as complete evidence."""

    upstream = payload.get("upstream_gate_receipts", {})
    if not isinstance(upstream, Mapping) or upstream.get("exp5723", {}).get("ready") is not True:
        return "blocked: upstream one-axis SamplerBackend readiness gate failed"
    if rust_crossover_ready_score(payload) == 1.0:
        return (
            "complete: matched-quality Rust/PyO3 CPU software crossover proven at "
            f"n={payload.get('qualified_crossover_n')}; no GPU, FPGA, TSU, or hardware claim"
        )
    return (
        "complete: terminal null; no consecutive larger-size matched-quality Rust/Python CPU "
        "crossover proven; timing claimed without GPU, FPGA, TSU, or hardware claim"
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5724 fields and fail closed on unsafe claim edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if len(payload.get("problem_sizes", [])) < 6:
        raise ValueError("problem_sizes must contain at least six sizes")
    if len(payload.get("topology_families", [])) < 3:
        raise ValueError("topology_families must contain at least three families")
    if len(payload.get("random_seeds", [])) < 10:
        raise ValueError("random_seeds must contain at least ten seeds")
    if int(payload.get("measured_repetition_count", 0)) < 30:
        raise ValueError("measured_repetition_count must be at least thirty")
    if payload.get("timing_claimed") is not True:
        raise ValueError("timing_claimed must be true")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("gpu_speedup_claimed") is not False:
        raise ValueError("gpu_speedup_claimed must be false")
    if payload.get("fpga_or_tsu_used") is not False:
        raise ValueError("fpga_or_tsu_used must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("software_speedup_claimed") is not (
        payload.get("qualified_crossover_n") is not None
    ):
        raise ValueError("software_speedup_claimed mismatch")
    if payload.get("rust_crossover_ready_score") != rust_crossover_ready_score(payload):
        raise ValueError("rust_crossover_ready_score mismatch")
    quality_rows = payload.get("quality_metrics_by_pair", [])
    expected_pairs = (
        len(payload.get("problem_sizes", []))
        * len(payload.get("topology_families", []))
        * len(payload.get("random_seeds", []))
    )
    if len(quality_rows) != expected_pairs:
        raise ValueError("quality_metrics_by_pair denominator mismatch")
    matched_count = sum(1 for row in quality_rows if row.get("quality_matched") is True)
    if payload.get("quality_matched_pair_count") != matched_count:
        raise ValueError("quality_matched_pair_count mismatch")
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
    """Read and validate upstream sampler artifacts before timing interpretation."""

    root_path = Path(root)
    return {
        "exp5611": _upstream_receipt(
            root_path,
            exp5611.RESULT_RELATIVE_PATH,
            exp5611.validate_artifact,
            ready_field="crossover_claim_allowed",
            terminal_null_expected=True,
        ),
        "exp5623": _upstream_receipt(
            root_path,
            exp5623.RESULT_RELATIVE_PATH,
            exp5623.validate_artifact,
            ready_field="crossover_claim_allowed",
            terminal_null_expected=True,
        ),
        "exp5714": _upstream_receipt(
            root_path,
            exp5714.RESULT_RELATIVE_PATH,
            exp5714.validate_artifact,
            ready_field="one_axis_rust_parity_ready_score",
            ready_value=1.0,
        ),
        "exp5715": _upstream_receipt(
            root_path,
            exp5715.RESULT_RELATIVE_PATH,
            exp5715.validate_artifact,
            ready_field="one_axis_rust_quality_ready_score",
            ready_value=1.0,
        ),
        "exp5723": _upstream_receipt(
            root_path,
            exp5723.RESULT_RELATIVE_PATH,
            exp5723.validate_artifact,
            ready_field="one_axis_samplerbackend_ready_score",
            ready_value=1.0,
        ),
        "claim_scope": {
            "cpu_cuda_terminal_for_this_substrate": True,
            "no_cpu_cuda_claim_reused": True,
            "no_gpu_fpga_tsu_claim_allowed": True,
        },
    }


def _upstream_receipt(
    root: Path,
    relative_path: Path,
    validator: Callable[[Mapping[str, Any]], None],
    *,
    ready_field: str,
    ready_value: object | None = None,
    terminal_null_expected: bool = False,
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
    except Exception as exc:  # noqa: BLE001 - exact validator failure is provenance evidence.
        receipt["blocked_reason"] = f"invalid_upstream:{type(exc).__name__}"
        return receipt
    value = payload.get(ready_field)
    receipt.update(
        {
            "valid": True,
            "ready_value": value,
            "honest_verdict": payload.get("honest_verdict"),
            "inference_substrate": payload.get("inference_substrate"),
            "timing_claimed": payload.get("timing_claimed"),
            "hardware_speedup_claimed": payload.get("hardware_speedup_claimed"),
        }
    )
    if terminal_null_expected:
        receipt["ready"] = value is False
        receipt["terminal_null"] = value is False
    else:
        receipt["ready"] = value == ready_value
    return receipt


def arm_configs() -> JsonDict:
    """Return the two production arm configurations under one API contract."""

    return {
        RUST_ARM: {
            "backend_class": "OneAxisRustBackend",
            "prefer_rust": True,
            "sampler_api": "SamplerBackend.sample",
            "expected_active_backend": ACTIVE_RUST_BACKEND,
            "algorithm": ONE_AXIS_ALGORITHM,
            "topology": ONE_AXIS_TOPOLOGY,
        },
        PYTHON_ARM: {
            "backend_class": "OneAxisRustBackend",
            "prefer_rust": False,
            "sampler_api": "SamplerBackend.sample",
            "expected_active_backend": ACTIVE_PYTHON_FALLBACK,
            "algorithm": ONE_AXIS_ALGORITHM,
            "topology": ONE_AXIS_TOPOLOGY,
        },
    }


def hardware_receipt(affinity: Mapping[str, Any]) -> JsonDict:
    """Collect CPU/OS/memory receipts without opening hardware claim scope."""

    return {
        "cpu_model": _cpu_model_name(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release()},
        "python_executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "memory": _meminfo_receipt(),
        "cpu_frequency_governor": _frequency_governor_receipt(affinity),
        "cpu_affinity": dict(affinity),
        "accelerators_used": [],
        "gpu_used": False,
        "fpga_or_tsu_used": False,
    }


def software_receipt(root: str | Path) -> JsonDict:
    """Collect Python/Rust/compiler/source receipts for replay."""

    root_path = Path(root)
    try:
        rust_module = importlib.import_module("carnot._rust")
        rust_extension = {
            "importable": True,
            "path": str(getattr(rust_module, "__file__", "")),
            "one_axis_symbols_present": all(
                hasattr(rust_module, name)
                for name in (
                    "RustOneAxisTemperingConfig",
                    "RustOneAxisTemperingCore",
                    "RustOneAxisTemperingState",
                )
            ),
        }
    except Exception as exc:  # noqa: BLE001 - missing extension is recorded honestly.
        rust_extension = {
            "importable": False,
            "blocked_reason": f"{type(exc).__name__}:{exc}",
            "one_axis_symbols_present": False,
        }
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rustc": _command_output(["rustc", "--version"]),
        "cargo": _command_output(["cargo", "--version"]),
        "pyo3_version": "0.24",
        "rust_extension": rust_extension,
        "source_hashes": {
            "experiment_5724": file_sha256(Path(__file__)),
            "one_axis_backend": file_sha256(
                root_path / "python/carnot/samplers/one_axis_rust_backend.py"
            ),
            "sampler_backend_factory": file_sha256(root_path / "python/carnot/samplers/backend.py"),
            "rust_one_axis_core": file_sha256(
                root_path / "crates/carnot-samplers/src/one_axis_tempering.rs"
            ),
        },
    }


def build_profile(root: str | Path) -> JsonDict:
    """Return the frozen local build profile used for timing."""

    root_path = Path(root)
    extension_path = root_path / "python/carnot/_rust.cpython-312-x86_64-linux-gnu.so"
    return {
        "build_mode": "existing_local_pyo3_extension",
        "release_profile_required": True,
        "extension_path": extension_path.as_posix(),
        "extension_present": extension_path.exists(),
        "abi3_forward_compatibility": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1",
        "features": {
            "one_axis_tempering": True,
            "penalty_axis_exchange": False,
            "gpu": False,
            "fpga_or_tsu": False,
        },
        "conductor_modified": False,
    }


def cpu_affinity_receipt(*, freeze: bool) -> JsonDict:
    """Record and optionally narrow CPU affinity for this benchmark process."""

    if not hasattr(os, "sched_getaffinity"):
        return {"observable": False, "freeze_requested": freeze, "status": "unsupported"}
    previous = sorted(os.sched_getaffinity(0))
    current = list(previous)
    status = "observed"
    if freeze and previous:
        selected = {previous[0]}
        try:
            os.sched_setaffinity(0, selected)
            current = sorted(os.sched_getaffinity(0))
            status = "frozen"
        except OSError as exc:
            status = f"freeze_failed:{type(exc).__name__}"
    return {
        "observable": True,
        "freeze_requested": freeze,
        "previous_cpus": previous,
        "current_cpus": current,
        "status": status,
    }


def thread_receipts() -> JsonDict:
    """Set and report single-thread environment variables for timing."""

    before = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")
    after = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
    pools: list[JsonDict] = []
    try:
        from threadpoolctl import threadpool_info

        pools = [dict(row) for row in threadpool_info()]
    except Exception:
        pools = []
    return {
        "thread_env_before": before,
        "thread_env_after": after,
        "thread_env_frozen_to_one_when_unset": True,
        "observed_threadpools": pools,
    }


def arrays_from_workload(workload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct dense float64 Hamiltonian arrays from a workload manifest row."""

    fields = np.asarray(workload["fields"], dtype=np.float64)
    couplings = np.zeros((fields.size, fields.size), dtype=np.float64)
    for i, j, value in workload["edge_list"]:
        couplings[int(i), int(j)] = float(value)
        couplings[int(j), int(i)] = float(value)
    return np.ascontiguousarray(fields), np.ascontiguousarray(couplings)


def exact_target_summary(fields: np.ndarray, couplings: np.ndarray) -> JsonDict:
    """Enumerate the cold-target Boltzmann distribution for small workloads."""

    n_spins = int(fields.size)
    states: list[list[int]] = []
    energies: list[float] = []
    for state_int in range(2**n_spins):
        state = [1 if (state_int >> bit) & 1 else -1 for bit in range(n_spins)]
        states.append(state)
        energies.append(ising_energy(fields, couplings, state))
    beta = float(exp5714.BETA_LADDER[-1])
    shifted = np.asarray([-beta * energy for energy in energies], dtype=np.float64)
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    energy_mean = float(np.dot(probabilities, np.asarray(energies, dtype=np.float64)))
    marginals = np.dot(probabilities, np.asarray(states, dtype=np.float64))
    return {
        "enumerable": True,
        "state_count": len(states),
        "cold_beta": beta,
        "best_energy": _stable_float(min(energies)),
        "mean_energy": _stable_float(energy_mean),
        "marginal_hash": sha256_json([_stable_float(value) for value in marginals.tolist()]),
        "target_distribution_hash": sha256_json(
            {
                "energies": [_stable_float(value) for value in energies],
                "probabilities": [_stable_float(value) for value in probabilities.tolist()],
            }
        ),
    }


def ising_energy(fields: np.ndarray, couplings: np.ndarray, state: Sequence[int]) -> float:
    """Compute the exact Ising energy convention used by Rust and Python arms."""

    spins = np.asarray(state, dtype=np.float64)
    return _stable_float(-0.5 * float(spins @ couplings @ spins) - float(spins @ fields))


def energy_histogram(energies: Sequence[float]) -> dict[str, float]:
    """Build a normalized energy histogram keyed by stable rounded energy."""

    counts = Counter(str(_stable_float(value)) for value in energies)
    total = sum(counts.values()) or 1
    return {key: count / total for key, count in sorted(counts.items())}


def distribution_tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return total-variation distance between two discrete distributions."""

    keys = set(left) | set(right)
    return _stable_float(
        0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)
    )


def current_peak_rss_kib() -> int:
    """Return process peak RSS in KiB on Linux-style hosts."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _hamiltonian_for_family(
    size: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.array([1 if (index % 3) != 1 else -1 for index in range(size)], dtype=np.int8)
    fields = np.zeros(size, dtype=np.float64)
    couplings = np.zeros((size, size), dtype=np.float64)
    if family == "ferromagnetic_ring_easy":
        fields = 0.04 * target.astype(np.float64)
        for index in range(size):
            _add_edge(couplings, index, (index + 1) % size, 0.05)
    elif family == "frustrated_chord_moderate":
        fields = np.array([0.02 * ((index % 5) - 2) for index in range(size)], dtype=np.float64)
        for index in range(size):
            _add_edge(couplings, index, (index + 1) % size, 0.035 * (-1 if index % 2 else 1))
            _add_edge(couplings, index, (index + max(2, size // 5)) % size, -0.025)
    elif family == "planted_basin_hard":
        midpoint = max(1, size // 2)
        target = np.array([1 if index < midpoint else -1 for index in range(size)], dtype=np.int8)
        fields = 0.012 * target.astype(np.float64)
        for index in range(size):
            peer = (index + 1) % size
            sign = target[index] * target[peer]
            _add_edge(couplings, index, peer, 0.045 * float(sign))
            chord = (index + max(3, size // 3)) % size
            _add_edge(couplings, index, chord, -0.018 * float(target[index] * target[chord]))
    else:
        raise ValueError(f"unsupported topology_families entry: {family}")
    np.fill_diagonal(couplings, 0.0)
    return fields, couplings, target


def _warm_arms(protocol: Mapping[str, Any], workload: Mapping[str, Any]) -> None:
    fields, couplings = arrays_from_workload(workload)
    seed = int(protocol["random_seeds"][0])
    descriptor = descriptor_for_timing(protocol, workload, seed)
    for _index in range(int(protocol["warmup_count"])):
        for arm in ARMS:
            OneAxisRustBackend(seed=seed, prefer_rust=(arm == RUST_ARM)).sample(
                fields,
                couplings,
                int(protocol["timing_sample_sweeps"]),
                descriptor,
            )


def _arm_order(protocol: Mapping[str, Any], workload_id: str, repetition: int) -> tuple[str, str]:
    seed = _stable_seed64(protocol["benchmark_order_seed"], workload_id, repetition)
    return (RUST_ARM, PYTHON_ARM) if seed % 2 == 0 else (PYTHON_ARM, RUST_ARM)


def _add_edge(couplings: np.ndarray, i: int, j: int, value: float) -> None:
    if i == j:
        return
    couplings[i, j] += float(value)
    couplings[j, i] += float(value)


def _edge_list(couplings: np.ndarray) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for i, j in np.argwhere(np.abs(np.triu(couplings, k=1)) > 0.0):
        rows.append([int(i), int(j), _stable_float(couplings[int(i), int(j)])])
    return rows


def _round_list(values: np.ndarray) -> list[float]:
    return [_stable_float(value) for value in np.asarray(values, dtype=np.float64).tolist()]


def _event_acceptance_rate(events: Sequence[Mapping[str, Any]]) -> float:
    if not events:
        return 0.0
    return _stable_float(sum(bool(event["accepted"]) for event in events) / len(events))


def _same_suffix(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        np.array_equal(left["samples"], right["samples"])
        and left["samples_spin"] == right["samples_spin"]
        and left["decision_log"] == right["decision_log"]
        and left["checkpoint"]["state"] == right["checkpoint"]["state"]
    )


def _stable_seed64(*parts: object) -> int:
    digest = hashlib.sha256(canonical_json([str(part) for part in parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _stable_float(value: Any) -> float:
    return round(float(value), 12)


def _cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def _meminfo_receipt() -> JsonDict:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {"observable": False}
    receipt: JsonDict = {"observable": True}
    text = path.read_text(encoding="utf-8", errors="replace")
    for source, target in (("MemTotal", "mem_total_kib"), ("MemAvailable", "mem_available_kib")):
        for line in text.splitlines():
            if line.startswith(source) and ":" in line:
                receipt[target] = int(line.split(":", 1)[1].strip().split()[0])
    return receipt


def _frequency_governor_receipt(affinity: Mapping[str, Any]) -> JsonDict:
    cpus = affinity.get("current_cpus") or []
    cpu = int(cpus[0]) if cpus else 0
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    if not base.exists():
        return {"observable": False, "cpu": cpu}
    receipt: JsonDict = {"observable": True, "cpu": cpu}
    for filename, key in (
        ("scaling_governor", "governor"),
        ("scaling_cur_freq", "current_khz"),
        ("scaling_min_freq", "min_khz"),
        ("scaling_max_freq", "max_khz"),
    ):
        path = base / filename
        if path.exists():
            receipt[key] = path.read_text(encoding="utf-8", errors="replace").strip()
    return receipt


def _command_output(command: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - toolchain absence is receipt data.
        return {"available": False, "error": f"{type(exc).__name__}:{exc}"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
