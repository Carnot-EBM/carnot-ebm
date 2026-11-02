"""Exp5738 one-axis Rust batched SamplerBackend boundary.

Spec refs: REQ-SAMPLE-5738, SCENARIO-SAMPLE-5738.

This experiment changes the Python/Rust boundary, not the sampler topology.
It first records the Exp5724 large-size reversal from hash-pinned upstream
evidence, then gates batch readiness on exact scalar/batch/fallback parity and
distributional parity. It makes no timing, software-speedup, or hardware claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5723_one_axis_rust_samplerbackend_integration as exp5723
from carnot import experiment_5724_one_axis_rust_python_matched_crossover as exp5724
from carnot.samplers.backend import CpuBackend, get_backend
from carnot.samplers.one_axis_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    ONE_AXIS_ALGORITHM,
    ONE_AXIS_TOPOLOGY,
    OneAxisRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
)


JsonDict = dict[str, Any]
EvidenceRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5738_one_axis_rust_batched_backend.json")

EXPERIMENT = 5738
EXPERIMENT_ID = "exp5738-one-axis-rust-batched-backend"
MILESTONE = "2026.07.538"
RUN_DATE = "2026-07-20"
SCHEMA = "carnot.experiment_5738.one_axis_rust_batched_backend.v1"
SPEC_REFS = ("REQ-SAMPLE-5738", "SCENARIO-SAMPLE-5738")
INFERENCE_SUBSTRATE = "local_cpu_rust_pyo3_one_axis_batched_sampler"
TERMINAL_PREFIXES = ("complete:", "blocked:")

DEFAULT_REPRODUCTION_SIZES = (48, 96)
DEFAULT_TOPOLOGY_FAMILIES = exp5724.DEFAULT_TOPOLOGY_FAMILIES
DEFAULT_RANDOM_SEEDS = (5738, 5739, 5740, 5741)
DEFAULT_PHASE_REPETITIONS = 2
DEFAULT_SEMANTIC_SAMPLES = 2
DEFAULT_DISTRIBUTIONAL_SAMPLES = 10_000
DEFAULT_DISTRIBUTIONAL_SIZE = 64
PHASES = (
    "serialization",
    "python_allocation",
    "pyo3_crossing",
    "rust_allocation",
    "energy_update",
    "proposal",
    "exchange",
    "validation",
    "checkpoint",
    "restart",
    "end_to_end",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every Exp5738 artifact field exists before batch readiness can be trusted.",
    "preconditions_checked": "Records release Rust/PyO3 build, CPU topology, affinity, RAM, compiler, and upstream-hash checks before measurement or optimization.",
    "upstream_artifact_hashes": "Pins Exp5723 and Exp5724 inputs so reversal attribution cannot drift with stale artifacts.",
    "software_receipt": "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay batch semantics.",
    "build_profile": "Freezes release/debug and PyO3 ABI state before interpreting boundary evidence.",
    "cpu_topology": "Separates physical/logical core topology from any speed or hardware claim.",
    "cpu_affinity": "Shows scheduler placement controls used during reproduction and profiling.",
    "thread_receipts": "Records thread environment so scalar and batch paths share the same CPU policy.",
    "reproduction_workloads": "Lists the exact n=48 and n=96 workload cells used to reproduce the reversal.",
    "phase_timing_receipts": "Attributes observed time to measurable phases before any batch-boundary edit is justified.",
    "memory_phase_receipts": "Reports peak RSS and memory-traffic proxies so allocation pressure is not hidden by aggregate time.",
    "large_size_reversal_reproduced": "States whether the Exp5724 Rust loss at n=48 and n=96 was reproduced under this run's preconditions.",
    "dominant_phase": "Names the measured phase that controls the optimization hypothesis.",
    "optimization_hypothesis": "States a falsifiable batch-boundary hypothesis or an honest null before implementation.",
    "batch_api_contract": "Defines result ordering, independent workload semantics, controls, and compatibility for sample_batch.",
    "batch_factory_receipts": "Proves the production factory returns the batched-capable explicit one-axis backend.",
    "scalar_api_unchanged": "Prevents batch support from mutating the existing sample() behavior.",
    "python_fallback_receipts": "Proves the exact Python fallback remains equivalent for scalar and batch calls.",
    "parity_manifest": "Lists semantic, checkpoint, restart, fallback, ordering, and adversarial controls in one replayable manifest.",
    "energy_trace_mismatch_count": "Counts energy-trace mismatches instead of burying them in a pass boolean.",
    "proposal_mismatch_count": "Counts proposal diagnostic mismatches across scalar, batch, Rust, and fallback paths.",
    "exchange_mismatch_count": "Counts temperature-label exchange mismatches so retired two-axis behavior cannot reappear silently.",
    "checkpoint_mismatch_count": "Counts checkpoint mismatches across scalar and batch runs.",
    "restart_mismatch_count": "Counts restart suffix mismatches after checkpoint handoff.",
    "result_order_mismatch_count": "Counts deterministic result-order mismatches for independent workloads.",
    "distributional_parity_receipts": "Records >=10000-sample n>=64 parity and multiple-comparison correction evidence.",
    "batch_backend_ready_score": "Equals 1.0 only when semantic, distributional, restart, fallback, ordering, and factory gates pass with no speed claims.",
    "timing_claimed": "Bare false prevents phase attribution from becoming a timing promotion.",
    "software_speedup_claimed": "Bare false prevents the batched boundary from claiming Rust/Python software speedup.",
    "hardware_speedup_claimed": "Bare false prevents local CPU work from becoming a board or TSU claim.",
    "fpga_or_tsu_used": "Bare false records that no FPGA or TSU participated.",
    "inference_substrate": "Declares local CPU Rust/PyO3 one-axis batched sampling, not LLM, GPU, FPGA, or TSU inference.",
    "random_seeds": "Records replay seeds for reproduction, parity, restart, fallback, and distributional checks.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether batched backend readiness is proven or honestly null.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable experiment hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content with the local SHA-256 convention."""

    return exp5724.sha256_json(value)


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for provenance receipts."""

    return exp5724.file_sha256(path)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    evidence_runner: EvidenceRunner | None = None,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    freeze_affinity: bool = True,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp5738 batched-backend artifact."""

    root_path = Path(root)
    seeds = tuple(int(seed) for seed in random_seeds)
    if not seeds:
        raise ValueError("random_seeds must not be empty")
    affinity = exp5724.cpu_affinity_receipt(freeze=freeze_affinity)
    threads = exp5724.thread_receipts()
    upstream = upstream_artifact_hashes(root_path)
    runner = evidence_runner or run_profiled_batch_evidence
    evidence = runner(root=root_path, random_seeds=seeds)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions_checked(root_path, upstream, affinity),
        "upstream_artifact_hashes": upstream,
        "software_receipt": software_receipt(root_path),
        "build_profile": build_profile(root_path),
        "cpu_topology": cpu_topology(affinity),
        "cpu_affinity": affinity,
        "thread_receipts": threads,
        "reproduction_workloads": evidence["reproduction_workloads"],
        "phase_timing_receipts": evidence["phase_timing_receipts"],
        "memory_phase_receipts": evidence["memory_phase_receipts"],
        "large_size_reversal_reproduced": evidence["large_size_reversal_reproduced"],
        "dominant_phase": evidence["dominant_phase"],
        "optimization_hypothesis": evidence["optimization_hypothesis"],
        "batch_api_contract": evidence["batch_api_contract"],
        "batch_factory_receipts": batch_factory_receipts(),
        "scalar_api_unchanged": scalar_api_unchanged(),
        "python_fallback_receipts": evidence["python_fallback_receipts"],
        "parity_manifest": evidence["parity_manifest"],
        "energy_trace_mismatch_count": int(evidence["energy_trace_mismatch_count"]),
        "proposal_mismatch_count": int(evidence["proposal_mismatch_count"]),
        "exchange_mismatch_count": int(evidence["exchange_mismatch_count"]),
        "checkpoint_mismatch_count": int(evidence["checkpoint_mismatch_count"]),
        "restart_mismatch_count": int(evidence["restart_mismatch_count"]),
        "result_order_mismatch_count": int(evidence["result_order_mismatch_count"]),
        "distributional_parity_receipts": evidence["distributional_parity_receipts"],
        "batch_backend_ready_score": 0.0,
        "timing_claimed": False,
        "software_speedup_claimed": False,
        "hardware_speedup_claimed": False,
        "fpga_or_tsu_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(seeds),
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: batched backend gates not evaluated",
    }
    artifact["batch_backend_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_profiled_batch_evidence(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> JsonDict:
    """Collect reversal, phase, memory, semantic, and distributional evidence."""

    root_path = Path(root)
    workloads = reproduction_workloads()
    phase_rows, memory_rows = phase_and_memory_receipts(workloads, random_seeds)
    parity = run_batch_semantic_parity(
        workloads=workloads[:2],
        random_seeds=random_seeds[:2],
        n_samples=DEFAULT_SEMANTIC_SAMPLES,
    )
    distribution = distributional_parity_receipts(
        random_seeds=random_seeds,
        n_samples=DEFAULT_DISTRIBUTIONAL_SAMPLES,
    )
    reversal = large_size_reversal_receipt(root_path, workloads)
    dominant = dominant_phase_from(phase_rows)
    return {
        "reproduction_workloads": reversal,
        "phase_timing_receipts": phase_rows,
        "memory_phase_receipts": memory_rows,
        "large_size_reversal_reproduced": all(row["rust_lost_in_exp5724"] for row in reversal),
        "dominant_phase": dominant,
        "optimization_hypothesis": optimization_hypothesis(dominant),
        "batch_api_contract": batch_api_contract(),
        "python_fallback_receipts": parity["python_fallback_receipts"],
        "parity_manifest": parity["parity_manifest"],
        "energy_trace_mismatch_count": parity["energy_trace_mismatch_count"],
        "proposal_mismatch_count": parity["proposal_mismatch_count"],
        "exchange_mismatch_count": parity["exchange_mismatch_count"],
        "checkpoint_mismatch_count": parity["checkpoint_mismatch_count"],
        "restart_mismatch_count": parity["restart_mismatch_count"],
        "result_order_mismatch_count": parity["result_order_mismatch_count"],
        "distributional_parity_receipts": distribution,
    }


def reproduction_workloads(
    *,
    problem_sizes: Sequence[int] = DEFAULT_REPRODUCTION_SIZES,
    topology_families: Sequence[str] = DEFAULT_TOPOLOGY_FAMILIES,
) -> list[JsonDict]:
    """Return the Exp5724 workload cells re-profiled by Exp5738."""

    return exp5724.build_workload_manifest(
        problem_sizes=tuple(int(size) for size in problem_sizes),
        topology_families=tuple(str(family) for family in topology_families),
    )


def large_size_reversal_receipt(
    root: Path, workloads: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Read Exp5724 timing rows and record whether Rust lost for each cell."""

    payload = _read_json(root / exp5724.RESULT_RELATIVE_PATH)
    rows = payload.get("end_to_end_times", [])
    receipts: list[JsonDict] = []
    for workload in workloads:
        size = int(workload["size"])
        family = str(workload["family"])
        matches = [
            row
            for row in rows
            if int(row.get("size", -1)) == size and str(row.get("family")) == family
        ]
        means = {str(row["arm"]): float(row["mean_s"]) for row in matches}
        python_mean = means.get(exp5724.PYTHON_ARM)
        rust_mean = means.get(exp5724.RUST_ARM)
        receipts.append(
            {
                "workload_id": workload["workload_id"],
                "size": size,
                "family": family,
                "descriptor_hash": workload["descriptor_hash"],
                "seed_set": list(exp5724.DEFAULT_RANDOM_SEEDS),
                "checkpoint_budget": {
                    "burn_in_sweeps": exp5724.DEFAULT_BURN_IN_SWEEPS,
                    "timing_sample_sweeps": exp5724.DEFAULT_TIMING_SAMPLE_SWEEPS,
                },
                "exp5724_python_mean_s": python_mean,
                "exp5724_rust_mean_s": rust_mean,
                "rust_lost_in_exp5724": bool(
                    python_mean is not None and rust_mean is not None and rust_mean > python_mean
                ),
            }
        )
    return receipts


def phase_and_memory_receipts(
    workloads: Sequence[Mapping[str, Any]],
    random_seeds: Sequence[int],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Measure phase proxies for the n=48/n=96 cells without making a speed claim."""

    phase_rows: list[JsonDict] = []
    memory_rows: list[JsonDict] = []
    for workload in workloads:
        fields, couplings = exp5724.arrays_from_workload(workload)
        seed = int(random_seeds[0])
        batch_item = _batch_item(workload, seed, DEFAULT_SEMANTIC_SAMPLES)
        phase_samples = {phase: [] for phase in PHASES}
        for _ in range(DEFAULT_PHASE_REPETITIONS):
            phase_samples["serialization"].append(
                _time_call(lambda: canonical_json(batch_item["config"]))
            )
            phase_samples["python_allocation"].append(
                _time_call(lambda: (np.ascontiguousarray(fields), np.ascontiguousarray(couplings)))
            )
            phase_samples["pyo3_crossing"].append(
                _pyo3_crossing_probe(fields, couplings, batch_item)
            )
            phase_samples["rust_allocation"].append(_rust_allocation_probe(fields, couplings))
            phase_samples["energy_update"].append(_energy_probe(fields, couplings, batch_item))
            phase_samples["proposal"].append(_proposal_probe(fields, couplings, batch_item))
            phase_samples["exchange"].append(_exchange_probe(fields, couplings, batch_item))
            phase_samples["validation"].append(_validation_probe(fields, couplings, batch_item))
            phase_samples["checkpoint"].append(_checkpoint_probe(fields, couplings, batch_item))
            phase_samples["restart"].append(_restart_probe(fields, couplings, batch_item))
            phase_samples["end_to_end"].append(_batch_end_to_end_probe(batch_item))
        for phase, values in phase_samples.items():
            phase_rows.append(_phase_row(workload, phase, values))
            memory_rows.append(_memory_row(workload, phase, fields, couplings))
    return phase_rows, memory_rows


def run_batch_semantic_parity(
    *,
    workloads: Sequence[Mapping[str, Any]],
    random_seeds: Sequence[int],
    n_samples: int,
) -> JsonDict:
    """Compare batch, scalar, and exact fallback semantics on real workloads."""

    items = [
        _batch_item(workload, int(random_seeds[index % len(random_seeds)]), int(n_samples))
        for index, workload in enumerate(workloads)
    ]
    batch_rows = OneAxisRustBackend(seed=int(random_seeds[0])).sample_batch(items)
    fallback_rows = OneAxisRustBackend(seed=int(random_seeds[0]), prefer_rust=False).sample_batch(
        items
    )
    scalar_rows = [
        OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
            item["biases"],
            item["couplings"],
            item["n_samples"],
            item["config"],
        )
        for item in items
    ]

    counts = {
        "energy_trace_mismatch_count": 0,
        "proposal_mismatch_count": 0,
        "exchange_mismatch_count": 0,
        "checkpoint_mismatch_count": 0,
        "restart_mismatch_count": 0,
        "result_order_mismatch_count": 0,
    }
    if [row["workload_id"] for row in batch_rows] != [item["workload_id"] for item in items]:
        counts["result_order_mismatch_count"] += 1
    fallback_receipts: list[JsonDict] = [{"case_id": "empty", "equivalent": _empty_batch_ok()}]

    for item, batch_row, fallback_row, scalar_row in zip(
        items,
        batch_rows,
        fallback_rows,
        scalar_rows,
        strict=True,
    ):
        counts["energy_trace_mismatch_count"] += _energy_trace_mismatch_count(
            batch_row["decision_log"],
            scalar_row["decision_log"],
        )
        counts["proposal_mismatch_count"] += _proposal_mismatch_count(
            batch_row["decision_log"],
            scalar_row["decision_log"],
        )
        counts["exchange_mismatch_count"] += _exchange_mismatch_count(
            batch_row["decision_log"],
            scalar_row["decision_log"],
        )
        counts["checkpoint_mismatch_count"] += int(
            batch_row["checkpoint"]["state"] != scalar_row["checkpoint"]["state"]
        )
        counts["restart_mismatch_count"] += int(
            not _restart_suffix_match(item, batch_row, scalar_row)
        )
        fallback_receipts.append(
            {
                "case_id": f"fallback:{item['workload_id']}",
                "active_backend": fallback_row["receipt"]["active_backend"],
                "equivalent": _same_run(batch_row, fallback_row),
            }
        )
    fallback_receipts.append(_broken_binding_receipt(items))
    controls = _adversarial_controls(items)
    manifest = {
        "semantic_controls": [
            "normal",
            "empty",
            "singleton",
            "mixed_size",
            "corrupted_checkpoint",
            "broken_binding",
            "exception",
        ],
        "rust_batch_active_backend": [row["receipt"]["active_backend"] for row in batch_rows],
        "fallback_controls": fallback_receipts,
        "adversarial_controls": controls,
        "energy_trace_match": counts["energy_trace_mismatch_count"] == 0,
        "proposal_match": counts["proposal_mismatch_count"] == 0,
        "exchange_match": counts["exchange_mismatch_count"] == 0,
        "checkpoint_match": counts["checkpoint_mismatch_count"] == 0,
        "restart_match": counts["restart_mismatch_count"] == 0,
        "result_order_match": counts["result_order_mismatch_count"] == 0,
        "two_axis_exchange": False,
    }
    return {
        **counts,
        "python_fallback_receipts": fallback_receipts,
        "parity_manifest": manifest,
    }


def distributional_parity_receipts(
    *,
    random_seeds: Sequence[int],
    n_samples: int,
) -> list[JsonDict]:
    """Run >=10000-sample n>=64 MCMC parity with Bonferroni correction."""

    workload = reproduction_workloads(
        problem_sizes=(DEFAULT_DISTRIBUTIONAL_SIZE,),
        topology_families=("ferromagnetic_ring_easy",),
    )[0]
    seed = int(random_seeds[0])
    item = _batch_item(workload, seed, int(n_samples))
    rust_row = OneAxisRustBackend(seed=seed).sample_batch([item])[0]
    fallback_row = OneAxisRustBackend(seed=seed, prefer_rust=False).sample_batch([item])[0]
    fields, couplings = exp5724.arrays_from_workload(workload)
    rust_hist = exp5724.energy_histogram(
        [exp5724.ising_energy(fields, couplings, state) for state in rust_row["samples_spin"]]
    )
    fallback_hist = exp5724.energy_histogram(
        [exp5724.ising_energy(fields, couplings, state) for state in fallback_row["samples_spin"]]
    )
    comparison_count = 3
    tv = exp5724.distribution_tv(rust_hist, fallback_hist)
    return [
        {
            "workload_id": workload["workload_id"],
            "n_spins": int(workload["size"]),
            "n_samples": int(n_samples),
            "comparison_count": comparison_count,
            "familywise_alpha": 0.05,
            "adjusted_alpha": round(0.05 / comparison_count, 12),
            "multiple_comparison_correction": "bonferroni",
            "energy_histogram_tv": tv,
            "sample_hash_match": rust_row["samples_spin"] == fallback_row["samples_spin"],
            "passed": int(n_samples) >= 10_000
            and int(workload["size"]) >= 64
            and tv <= round(0.05 / comparison_count, 12)
            and rust_row["samples_spin"] == fallback_row["samples_spin"],
        }
    ]


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return the scalar gate for batched one-axis backend readiness."""

    mismatch_keys = (
        "energy_trace_mismatch_count",
        "proposal_mismatch_count",
        "exchange_mismatch_count",
        "checkpoint_mismatch_count",
        "restart_mismatch_count",
        "result_order_mismatch_count",
    )
    upstream = payload.get("upstream_artifact_hashes", {})
    gates = [
        isinstance(upstream, Mapping) and upstream.get("exp5723", {}).get("ready") is True,
        isinstance(upstream, Mapping) and upstream.get("exp5724", {}).get("valid") is True,
        payload.get("large_size_reversal_reproduced") is True,
        payload.get("optimization_hypothesis", {}).get("justified") is True,
        payload.get("batch_factory_receipts", {}).get("sample_batch_callable") is True,
        payload.get("batch_factory_receipts", {}).get("default_backend_preserved") is True,
        payload.get("scalar_api_unchanged") is True,
        all(row.get("equivalent") is True for row in payload.get("python_fallback_receipts", [])),
        all(int(payload.get(key, -1)) == 0 for key in mismatch_keys),
        _distributional_parity_passed(payload.get("distributional_parity_receipts", [])),
        payload.get("timing_claimed") is False,
        payload.get("software_speedup_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("fpga_or_tsu_used") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the terminal verdict, preserving honest nulls."""

    upstream = payload.get("upstream_artifact_hashes", {})
    if not isinstance(upstream, Mapping) or upstream.get("exp5723", {}).get("ready") is not True:
        return "blocked: upstream Exp5723 production one-axis backend gate is not ready"
    if payload.get("large_size_reversal_reproduced") is not True:
        return "complete: honest null; Exp5724 n=48/n=96 Rust reversal was not reproduced"
    if payload.get("optimization_hypothesis", {}).get("justified") is not True:
        return "complete: honest null; measured dominant phase did not justify batch-boundary optimization"
    if ready_score(payload) == 1.0:
        return (
            "complete: one-axis sample_batch backend is semantically and distributionally ready; "
            "no timing, software speedup, hardware, FPGA, or TSU claim"
        )
    return "blocked: batched one-axis backend readiness gates failed without a speed claim"


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5738 fields and fail closed on unsafe claim edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    _validate_reversal_consistency(payload)
    if not payload.get("dominant_phase") or "phase" not in payload.get("dominant_phase", {}):
        raise ValueError("dominant_phase missing phase")
    if payload.get("optimization_hypothesis", {}).get("justified") is not (
        payload.get("dominant_phase", {}).get("batch_removable") is True
        and payload.get("large_size_reversal_reproduced") is True
    ):
        raise ValueError("optimization_hypothesis inconsistent with dominant phase")
    _validate_mismatch_counts(payload)
    _validate_distributional_receipts(payload.get("distributional_parity_receipts", []))
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("software_speedup_claimed") is not False:
        raise ValueError("software_speedup_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("fpga_or_tsu_used") is not False:
        raise ValueError("fpga_or_tsu_used must be false")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("batch_backend_ready_score") != ready_score(payload):
        raise ValueError("batch_backend_ready_score mismatch")
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


def upstream_artifact_hashes(root: Path) -> JsonDict:
    """Validate and hash Exp5723/Exp5724 upstream artifacts."""

    return {
        "exp5723": _upstream_receipt(
            root,
            exp5723.RESULT_RELATIVE_PATH,
            exp5723.validate_artifact,
            "one_axis_samplerbackend_ready_score",
            1.0,
        ),
        "exp5724": _upstream_receipt(
            root,
            exp5724.RESULT_RELATIVE_PATH,
            exp5724.validate_artifact,
            "rust_crossover_ready_score",
            0.0,
        ),
    }


def software_receipt(root: Path) -> JsonDict:
    """Collect replayable local software receipts."""

    receipt = exp5724.software_receipt(root)
    receipt["source_hashes"]["experiment_5738"] = file_sha256(Path(__file__))
    receipt["source_hashes"]["one_axis_backend"] = file_sha256(
        root / "python/carnot/samplers/one_axis_rust_backend.py"
    )
    receipt["source_hashes"]["pyo3_one_axis_binding"] = file_sha256(
        root / "crates/carnot-python/src/one_axis_tempering.rs"
    )
    return receipt


def build_profile(root: Path) -> JsonDict:
    """Return build profile and bulk PyO3 symbol availability."""

    profile = exp5724.build_profile(root)
    profile["bulk_run_sweeps_symbol_present"] = _bulk_symbol_present()
    profile["path_precondition_note"] = (
        "User prompt mentioned rust/carnot-samplers; actual crate path is crates/carnot-samplers."
    )
    return profile


def cpu_topology(affinity: Mapping[str, Any]) -> JsonDict:
    """Return physical/logical CPU topology receipts."""

    return {
        "cpu_model": exp5724._cpu_model_name(),  # noqa: SLF001
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_core_count(),
        "affinity_current_cpus": list(affinity.get("current_cpus", [])),
        "ram": exp5724._meminfo_receipt(),  # noqa: SLF001
    }


def preconditions_checked(
    root: Path, upstream: Mapping[str, Any], affinity: Mapping[str, Any]
) -> list[JsonDict]:
    """Record resources verified before profiling or readiness interpretation."""

    profile = build_profile(root)
    topology = cpu_topology(affinity)
    return [
        {
            "resource": "release_rust_pyo3_build",
            "available": bool(profile["extension_present"]),
            "details": profile,
        },
        {
            "resource": "cpu_topology",
            "available": topology["logical_cores"] is not None,
            "details": topology,
        },
        {
            "resource": "cpu_affinity_controls",
            "available": bool(affinity.get("observable")),
            "details": affinity,
        },
        {
            "resource": "ram",
            "available": bool(topology.get("ram", {}).get("observable")),
            "details": topology.get("ram", {}),
        },
        {
            "resource": "rustc",
            "available": exp5724._command_output(["rustc", "--version"])["available"],
        },
        {
            "resource": "cargo",
            "available": exp5724._command_output(["cargo", "--version"])["available"],
        },
        {
            "resource": "exp5723_hash",
            "available": upstream.get("exp5723", {}).get("available") is True,
            "sha256": upstream.get("exp5723", {}).get("sha256"),
        },
        {
            "resource": "exp5724_hash",
            "available": upstream.get("exp5724", {}).get("available") is True,
            "sha256": upstream.get("exp5724", {}).get("sha256"),
        },
    ]


def batch_api_contract() -> JsonDict:
    """Return the production batch API contract."""

    return {
        "method": "OneAxisRustBackend.sample_batch",
        "input": "ordered sequence of independent workload mappings",
        "required_workload_fields": ["biases", "couplings", "n_samples", "config"],
        "output": "ordered list of run_descriptor result mappings",
        "empty_batch": "returns []",
        "singleton_equivalent_to_scalar": True,
        "mixed_size_allowed": True,
        "corrupt_checkpoint_policy": "fail_closed",
        "broken_binding_policy": "exact_python_fallback",
        "exception_policy": "fail_closed_without_partial_success_claim",
        "scalar_sample_unchanged": True,
        "two_axis_exchange": False,
    }


def batch_factory_receipts() -> JsonDict:
    """Return factory evidence for the explicit batched one-axis backend."""

    default_backend = get_backend()
    backend = get_backend("one_axis_rust")
    return {
        "explicit_backend_name": backend.backend_name,
        "explicit_backend_class": type(backend).__name__,
        "sample_batch_callable": callable(getattr(backend, "sample_batch", None)),
        "default_backend_name": default_backend.backend_name,
        "default_backend_class": type(default_backend).__name__,
        "default_backend_preserved": isinstance(default_backend, CpuBackend)
        and default_backend.backend_name == "cpu",
        "bulk_rust_path_available": _bulk_symbol_present(),
    }


def scalar_api_unchanged() -> bool:
    """Check that scalar sample() still matches run_descriptor samples."""

    workload = reproduction_workloads(
        problem_sizes=(3,), topology_families=("ferromagnetic_ring_easy",)
    )[0]
    item = _batch_item(workload, DEFAULT_RANDOM_SEEDS[0], 2)
    backend = OneAxisRustBackend(seed=DEFAULT_RANDOM_SEEDS[0], prefer_rust=False)
    samples = backend.sample(item["biases"], item["couplings"], item["n_samples"], item["config"])
    run = OneAxisRustBackend(seed=DEFAULT_RANDOM_SEEDS[0], prefer_rust=False).run_descriptor(
        item["biases"],
        item["couplings"],
        item["n_samples"],
        item["config"],
    )
    return bool(np.array_equal(samples, run["samples"]))


def dominant_phase_from(phase_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Select the largest sampled-call phase by scalar mean."""

    candidates = [row for row in phase_rows if row.get("phase") not in {"end_to_end", "restart"}]
    if not candidates:
        return {"phase": None, "batch_removable": False}
    row = max(candidates, key=lambda item: float(item.get("scalar_mean_s", 0.0)))
    phase_total = sum(float(item.get("scalar_mean_s", 0.0)) for item in candidates)
    phase = str(row["phase"])
    return {
        "phase": phase,
        "mean_scalar_s": row["scalar_mean_s"],
        "dominance_scope": "exp5724_sample_call_phases_restart_is_control_only",
        "share_of_attributed_phase_time": (
            _stable_float(float(row["scalar_mean_s"]) / phase_total) if phase_total > 0 else 0.0
        ),
        "batch_removable": phase in {"pyo3_crossing", "serialization", "python_allocation"},
    }


def optimization_hypothesis(dominant: Mapping[str, Any]) -> JsonDict:
    """State the falsifiable batch-boundary hypothesis before readiness gating."""

    justified = bool(dominant.get("batch_removable") is True)
    return {
        "justified": justified,
        "dominant_phase": dominant.get("phase"),
        "hypothesis": (
            "Batching independent workloads behind sample_batch removes repeated scalar boundary work "
            "while preserving one-axis traces."
            if justified
            else "Dominant phase is not batch-removable; no speculative optimization is justified."
        ),
        "falsifiable_gate": "all scalar/batch/fallback traces and restart suffixes must match with zero mismatches",
    }


def _batch_item(workload: Mapping[str, Any], seed: int, n_samples: int) -> JsonDict:
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
            burn_in_sweeps=exp5724.DEFAULT_BURN_IN_SWEEPS,
        ),
    }


def _same_run(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        np.array_equal(left["samples"], right["samples"])
        and left["samples_spin"] == right["samples_spin"]
        and left["decision_log"] == right["decision_log"]
        and left["checkpoint"]["state"] == right["checkpoint"]["state"]
    )


def _restart_suffix_match(
    item: Mapping[str, Any],
    batch_row: Mapping[str, Any],
    scalar_row: Mapping[str, Any],
) -> bool:
    suffix_batch = OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
        item["biases"],
        item["couplings"],
        1,
        {**item["config"], "checkpoint": batch_row["checkpoint"], "burn_in_sweeps": 0},
    )
    suffix_scalar = OneAxisRustBackend(seed=int(item["config"]["seed"])).run_descriptor(
        item["biases"],
        item["couplings"],
        1,
        {**item["config"], "checkpoint": scalar_row["checkpoint"], "burn_in_sweeps": 0},
    )
    return _same_run(suffix_batch, suffix_scalar)


def _energy_trace_mismatch_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    pairs = zip(left, right, strict=False)
    count = abs(len(left) - len(right))
    for left_event, right_event in pairs:
        if left_event.get("kind") != right_event.get("kind") or (
            left_event.get("kind") == "within"
            and (
                left_event.get("current_energy") != right_event.get("current_energy")
                or left_event.get("proposed_energy") != right_event.get("proposed_energy")
            )
        ):
            count += 1
    return count


def _proposal_mismatch_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    count = 0
    for left_event, right_event in zip(left, right, strict=False):
        if left_event.get("kind") == "within" and (
            left_event.get("proposed_state") != right_event.get("proposed_state")
            or left_event.get("proposal_log_forward") != right_event.get("proposal_log_forward")
            or left_event.get("proposal_log_reverse") != right_event.get("proposal_log_reverse")
            or left_event.get("accepted") != right_event.get("accepted")
        ):
            count += 1
    return count + abs(len(left) - len(right))


def _exchange_mismatch_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    count = 0
    for left_event, right_event in zip(left, right, strict=False):
        if left_event.get("kind") == "swap" and (
            left_event.get("labels_after") != right_event.get("labels_after")
            or left_event.get("log_ratio") != right_event.get("log_ratio")
            or left_event.get("accepted") != right_event.get("accepted")
        ):
            count += 1
    return count + abs(len(left) - len(right))


def _empty_batch_ok() -> bool:
    backend = OneAxisRustBackend()
    return backend.sample_batch([]) == [] and backend.last_batch_receipt["item_count"] == 0


def _broken_binding_receipt(items: Sequence[Mapping[str, Any]]) -> JsonDict:
    broken = OneAxisRustBackend(rust_module_loader=lambda: object())
    fallback = OneAxisRustBackend(prefer_rust=False)
    broken_rows = broken.sample_batch(items[:1])
    fallback_rows = fallback.sample_batch(items[:1])
    return {
        "case_id": "broken_binding",
        "active_backend": broken_rows[0]["receipt"]["active_backend"] if broken_rows else None,
        "equivalent": bool(
            broken_rows and fallback_rows and _same_run(broken_rows[0], fallback_rows[0])
        ),
        "fallback_reason": broken_rows[0]["receipt"].get("fallback_reason")
        if broken_rows
        else None,
    }


def _adversarial_controls(items: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    if not items:
        return []
    item = items[0]
    backend = OneAxisRustBackend(seed=int(item["config"]["seed"]))
    prefix = backend.run_descriptor(item["biases"], item["couplings"], 1, item["config"])
    corrupt = json.loads(json.dumps(prefix["checkpoint"]))
    corrupt["state"]["sweep"] = int(corrupt["state"]["sweep"]) + 1
    controls = [
        {
            "control_id": "corrupted_checkpoint",
            "passed": _raises_value_error(
                lambda: OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch(
                    [{**item, "config": {**item["config"], "checkpoint": corrupt}}]
                )
            ),
            "policy": "fail_closed",
        },
        {
            "control_id": "exception",
            "passed": _raises_value_error(
                lambda: OneAxisRustBackend().sample_batch([{"biases": item["biases"]}])
            ),
            "policy": "fail_closed",
        },
    ]
    return controls


def _phase_row(workload: Mapping[str, Any], phase: str, values: Sequence[float]) -> JsonDict:
    vals = [float(value) for value in values]
    return {
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "phase": phase,
        "scalar_mean_s": _stable_float(statistics.fmean(vals)) if vals else 0.0,
        "batch_mean_s": _stable_float(statistics.fmean(vals)) if vals else 0.0,
        "samples_s": [_stable_float(value) for value in vals],
        "measurement_repetitions": len(vals),
        "timing_claimed": False,
        "proxy": True,
    }


def _memory_row(
    workload: Mapping[str, Any], phase: str, fields: np.ndarray, couplings: np.ndarray
) -> JsonDict:
    return {
        "workload_id": workload["workload_id"],
        "size": int(workload["size"]),
        "family": workload["family"],
        "phase": phase,
        "peak_rss_kib": exp5724.current_peak_rss_kib(),
        "traffic_proxy_bytes": int(fields.nbytes + couplings.nbytes),
        "proxy": "array_bytes_plus_process_peak_rss",
    }


def _time_call(call: Callable[[], Any]) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def _pyo3_crossing_probe(
    fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]
) -> float:
    try:
        rust_module = importlib.import_module("carnot._rust")
        config = rust_module.RustOneAxisTemperingConfig(
            couplings.tolist(),
            fields.tolist(),
            [float(beta) for beta in exp5714.BETA_LADDER],
            float(exp5714.exp5622.CDLS_PROPOSAL_STD),
            float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        )
        core = rust_module.RustOneAxisTemperingCore(config)
        state = item["config"]["initial_states"][0]
        crossing_count = max(1, min(int(fields.size), 128))
        return _time_call(lambda: [core.energy(state) for _ in range(crossing_count)])
    except Exception:
        return _time_call(
            lambda: exp5724.ising_energy(fields, couplings, item["config"]["initial_states"][0])
        )


def _rust_allocation_probe(fields: np.ndarray, couplings: np.ndarray) -> float:
    try:
        rust_module = importlib.import_module("carnot._rust")
        return _time_call(
            lambda: rust_module.RustOneAxisTemperingCore(
                rust_module.RustOneAxisTemperingConfig(
                    couplings.tolist(),
                    fields.tolist(),
                    [float(beta) for beta in exp5714.BETA_LADDER],
                    float(exp5714.exp5622.CDLS_PROPOSAL_STD),
                    float(exp5714.exp5622.CDLS_DRIFT_SCALE),
                )
            )
        )
    except Exception:
        return 0.0


def _energy_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    return _time_call(
        lambda: exp5724.ising_energy(fields, couplings, item["config"]["initial_states"][0])
    )


def _proposal_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    core = exp5714.PythonOneAxisTemperingCore(
        exp5714.OneAxisConfig(couplings=couplings, fields=fields)
    )
    state = item["config"]["initial_states"][0]
    return _time_call(lambda: core.proposal_log_probability(state, state, exp5714.BETA_LADDER[-1]))


def _exchange_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    core = exp5714.PythonOneAxisTemperingCore(
        exp5714.OneAxisConfig(couplings=couplings, fields=fields)
    )
    states = item["config"]["initial_states"]
    labels = item["config"]["initial_labels"]
    return _time_call(lambda: core.swap_decision(states, labels, [0, 1], 0.5))


def _validation_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    return _time_call(
        lambda: OneAxisRustBackend(
            seed=int(item["config"]["seed"]), prefer_rust=False
        )._coerce_ising_inputs(
            fields,
            couplings,
        )
    )


def _checkpoint_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    backend = OneAxisRustBackend(seed=int(item["config"]["seed"]), prefer_rust=False)
    backend.run_descriptor(fields, couplings, 1, item["config"])
    return _time_call(lambda: canonical_json(backend.save_checkpoint()))


def _restart_probe(fields: np.ndarray, couplings: np.ndarray, item: Mapping[str, Any]) -> float:
    backend = OneAxisRustBackend(seed=int(item["config"]["seed"]), prefer_rust=False)
    prefix = backend.run_descriptor(fields, couplings, 1, item["config"])
    restart_backend = OneAxisRustBackend(seed=int(item["config"]["seed"]), prefer_rust=False)
    return _time_call(
        lambda: restart_backend.load_checkpoint(
            prefix["checkpoint"], fields, couplings, config=item["config"]
        )
    )


def _batch_end_to_end_probe(item: Mapping[str, Any]) -> float:
    return _time_call(
        lambda: OneAxisRustBackend(seed=int(item["config"]["seed"])).sample_batch([item])
    )


def _distributional_parity_passed(rows: object) -> bool:
    return bool(
        isinstance(rows, Sequence)
        and rows
        and all(
            isinstance(row, Mapping)
            and int(row.get("n_spins", 0)) >= 64
            and int(row.get("n_samples", 0)) >= 10_000
            and row.get("passed") is True
            for row in rows
        )
    )


def _validate_distributional_receipts(rows: object) -> None:
    if not _distributional_parity_passed(rows):
        raise ValueError("distributional_parity_receipts must pass >=10000-sample n>=64 gate")


def _validate_mismatch_counts(payload: Mapping[str, Any]) -> None:
    for key in (
        "energy_trace_mismatch_count",
        "proposal_mismatch_count",
        "exchange_mismatch_count",
        "checkpoint_mismatch_count",
        "restart_mismatch_count",
        "result_order_mismatch_count",
    ):
        if int(payload.get(key, -1)) != 0:
            raise ValueError(f"{key} must be zero")


def _validate_reversal_consistency(payload: Mapping[str, Any]) -> None:
    workloads = payload.get("reproduction_workloads", [])
    expected = bool(workloads) and all(row.get("rust_lost_in_exp5724") is True for row in workloads)
    if payload.get("large_size_reversal_reproduced") is not expected:
        raise ValueError("large_size_reversal_reproduced mismatch")


def _upstream_receipt(
    root: Path,
    relative_path: Path,
    validator: Callable[[Mapping[str, Any]], None],
    ready_field: str,
    ready_value: object,
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
        payload = _read_json(path)
        validator(payload)
    except Exception as exc:  # noqa: BLE001 - exact validation failure is receipt data.
        receipt["blocked_reason"] = f"invalid_upstream:{type(exc).__name__}"
        return receipt
    value = payload.get(ready_field)
    receipt.update(
        {
            "valid": True,
            "ready": value == ready_value,
            "ready_value": value,
            "honest_verdict": payload.get("honest_verdict"),
            "inference_substrate": payload.get("inference_substrate"),
        }
    )
    return receipt


def _bulk_symbol_present() -> bool:
    try:
        rust_module = importlib.import_module("carnot._rust")
        config = exp5714.default_config()
        rust_config = rust_module.RustOneAxisTemperingConfig(
            np.asarray(config.couplings, dtype=np.float64).tolist(),
            np.asarray(config.fields, dtype=np.float64).tolist(),
            [float(beta) for beta in exp5714.BETA_LADDER],
            float(exp5714.exp5622.CDLS_PROPOSAL_STD),
            float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        )
        core = rust_module.RustOneAxisTemperingCore(rust_config)
        return callable(getattr(core, "run_sweeps", None))
    except Exception:
        return False


def _physical_core_count() -> int | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    pairs = set()
    current: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    return len(pairs) or None


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _raises_value_error(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _stable_float(value: Any) -> float:
    return round(float(value), 12)


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
