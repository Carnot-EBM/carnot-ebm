"""Exp5623 corrected-cDLS multi-seed CPU/CUDA crossover benchmark.

Spec refs: REQ-SAMPLE-5623, SCENARIO-SAMPLE-5623.

The experiment is deliberately conservative. Exp5622 first proves that the
continuous-intermediate cDLS projection has an exact discrete Metropolis-
Hastings correction on enumerable Ising targets. This module refuses to run the
large CPU/CUDA timing loop unless that upstream receipt is ready and checksum-
valid. Timing rows are useful only after quality is frozen, so every size/seed
pair receives an explicit inclusion or exclusion explanation before any speedup
summary is allowed to use it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from math import exp, sqrt
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot import experiment_5573_matched_sampler_hardware_continuity as exp5573
from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622


JsonDict = dict[str, Any]
Clock = Callable[[], float]
IsingInstance = exp5573.IsingInstance
SamplerRunner = Callable[
    [list[IsingInstance], tuple[int, ...], int, JsonDict, Any, Clock, float | None],
    tuple[list[JsonDict], list[JsonDict]],
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json")
SUFFICIENT_STATISTICS_RELATIVE_PATH = Path(
    "results/experiment_5623_cdls_multiseed_cpu_cuda_crossover_sufficient_statistics.json"
)
UPSTREAM_GATE_RELATIVE_PATH = exp5622.RESULT_RELATIVE_PATH

EXPERIMENT = 5623
EXPERIMENT_ID = "exp5623-cdls-multiseed-cpu-cuda-crossover"
MILESTONE = "2026.07.507"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5623.cdls_multiseed_cpu_cuda_crossover.v1"
SPEC_REFS = ("REQ-SAMPLE-5623", "SCENARIO-SAMPLE-5623")
INFERENCE_SUBSTRATE = "matched_cpu_cuda_exact_ising_sampling"

DISCRETE_METHOD = "discrete_dls_heat_bath"
CORRECTED_CDLS_METHOD = "corrected_cdls_projection_mh"
FORBIDDEN_METHODS = {"uncorrected_cdls_projection", "biased_temperature_positive_control"}
METHOD_IDS = (DISCRETE_METHOD, CORRECTED_CDLS_METHOD)
DEVICES = ("cpu", "cuda")

DEFAULT_INSTANCE_SIZES = (128, 256, 512, 1024)
DEFAULT_SEEDS = (5623, 5624, 5625, 5626, 5627)
DEFAULT_SAMPLES_PER_PAIR = 10_000
DEFAULT_WARMUP_STEPS = exp5573.DEFAULT_WARMUP_STEPS
DEFAULT_THINNING = exp5573.DEFAULT_THINNING
DEFAULT_TEMPERATURE = exp5573.DEFAULT_TEMPERATURE
DEFAULT_PRECISION = exp5573.DEFAULT_PRECISION
DEFAULT_STOPPING_RULE = exp5573.DEFAULT_STOPPING_RULE
DEFAULT_ROW_TIMEOUT_S = 12.0
MIN_SAMPLES_PER_SUCCESS = 10_000
MIN_PAIRED_SEEDS = 5
TERMINAL_PREFIXES = ("complete:", "blocked:")

QUALITY_THRESHOLDS: JsonDict = {
    "energy_histogram_tv_delta_max": 0.03,
    "mean_energy_worse_abs_max": 0.5,
    "mean_energy_worse_rel_max": 0.05,
    "best_energy_worse_abs_max": 0.5,
    "constraint_satisfaction_rate_drop_max": 0.03,
    "min_effective_sample_size": 100.0,
    "max_integrated_autocorrelation_ratio": 2.0,
    "min_corrected_cdls_acceptance_rate": 0.01,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every headline and gate field exists before a reviewer trusts the JSON shape.",
    "upstream_gate_receipt": "Proves Exp5622 exactness prerequisites are fixed before GPU work is allocated.",
    "target_descriptors": "Proves both methods sample identical exact Ising Hamiltonians.",
    "instance_sizes": "Makes the tested crossover range explicit instead of implying unmeasured scale.",
    "models_tested": "Names sampler methods only, keeping the unchanged discrete baseline and corrected cDLS kernel separate with no LLM implied.",
    "seeds": "Records at least five paired seeds so every CPU/CUDA and method comparison is reproducible.",
    "samples_per_pair": "Guards the >=10000 retained-sample floor required for mixing estimates.",
    "cpu_device_receipt": "Authenticates CPU identity, runtime, and free memory for the local benchmark.",
    "cuda_device_receipt": "Authenticates CUDA identity, driver/runtime, and free memory or records a precise blocker.",
    "timing_rows": "Preserves raw matched timing evidence, including failed rows, before any summary ratio.",
    "energy_quality_metrics": "Prevents speed from hiding wrong samples by reporting energy and exact constraint quality.",
    "mixing_metrics": "Reports ESS and autocorrelation to test whether the corrected cDLS mechanism actually mixes.",
    "quality_gate_results_by_pair": "Explains every pair's inclusion or exclusion before timing enters a speedup claim.",
    "successful_matched_pairs": "Lists only quality-matched complete method/device pairs; failed rows do not enter speedups.",
    "speedup_by_pair": "Reports only matched ratios with intervals; failed or quality-inferior rows cannot enter speedups.",
    "timing_intervals_by_size": "Aggregates paired-seed ratios into intervals so a crossover cannot rest on one lucky row.",
    "sufficient_statistics_path": "Points to recomputable energy, constraint, and timing evidence instead of trusting summaries.",
    "sufficient_statistics_sha256": "Pins the sufficient-statistic file used to recompute ESS, autocorrelation, quality, and timing.",
    "crossover_size": "Records the smallest gated crossover size, or null when no crossover is proven.",
    "crossover_claim_allowed": "Requires quality and timing gates to pass together before any crossover claim.",
    "board_speedup_claimed": "Bare false prevents this CPU/CUDA sampler study from reopening board or TSU scope.",
    "inference_substrate": "Declares matched CPU/CUDA exact Ising sampling, not LLM inference or board timing.",
    "random_seeds": "Duplicates the paired seeds in methodology form for verifier compatibility.",
    "reproducibility_checksum": "Content-addresses the benchmark artifact and sufficient-statistic receipt.",
    "honest_verdict": "Terminal complete: or blocked: verdict states whether no-crossover evidence is final.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes are stable across reruns."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Return the repository-standard SHA-256 hex digest for text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible values after canonical serialization."""

    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    """Hash a file byte-for-byte for artifact receipts."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def read_json(path: Path) -> JsonDict:
    """Load a JSON object and reject non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def row_id(instance_id: str, seed: int, device: str, method_id: str) -> str:
    """Build the stable row identifier used by timing and sufficient stats."""

    return f"{instance_id}:seed{int(seed)}:{device}:{method_id}"


def corrected_cdls_kernel_parameters() -> JsonDict:
    """Return the exact cDLS constants inherited from Exp5622.

    These constants are intentionally not retuned here. The point of Exp5623 is
    to time the kernel whose correctness was already audited, not to search for
    a faster or friendlier proposal after seeing large-n behavior.
    """

    return {
        "final_kernel": CORRECTED_CDLS_METHOD,
        "proposal_std": exp5622.CDLS_PROPOSAL_STD,
        "drift_scale": exp5622.CDLS_DRIFT_SCALE,
        "continuous_bound": exp5622.CDLS_CONTINUOUS_BOUND,
        "large_n_timing_tuned": False,
        "biased_control_kernel_used": False,
    }


def models_tested() -> list[JsonDict]:
    """Name the two sampler methods; no language model is part of this test."""

    return [
        {
            "model_id": DISCRETE_METHOD,
            "role": "unchanged_discrete_langevin_baseline",
            "description": "Single-site heat-bath discrete Ising baseline using the existing Exp5573 transition rule.",
            "baseline_preserved": True,
            "biased_control_kernel_used": False,
            "llm_involved": False,
        },
        {
            "model_id": CORRECTED_CDLS_METHOD,
            "role": "corrected_continuous_intermediate_discrete_langevin",
            "description": "Exp5622 final projected cDLS proposal with exact discrete Metropolis-Hastings correction.",
            "baseline_preserved": False,
            "projection_correction": "metropolis_hastings_exact_projected_proposal",
            "kernel_parameters": corrected_cdls_kernel_parameters(),
            "biased_control_kernel_used": False,
            "llm_involved": False,
        },
    ]


def load_upstream_gate_receipt(root: str | Path = REPO_ROOT) -> JsonDict:
    """Authenticate Exp5622 before the benchmark is allowed to allocate GPU work."""

    path = Path(root) / UPSTREAM_GATE_RELATIVE_PATH
    receipt: JsonDict = {
        "path": UPSTREAM_GATE_RELATIVE_PATH.as_posix(),
        "available": path.exists(),
        "sha256": None,
        "ready": False,
        "blocked_reason": None,
        "schema": None,
        "experiment": None,
        "run_date": None,
        "kernel_audit_ready_score": None,
        "final_kernel": None,
        "correction_applied": None,
        "broken_kernel_controls_rejected": None,
        "large_n_timing_tuned": None,
        "quality_gate_specification": [],
    }
    if not path.exists():
        receipt["blocked_reason"] = "upstream_gate_missing"
        return receipt
    receipt["sha256"] = file_sha256(path)
    try:
        payload = read_json(path)
        exp5622.validate_artifact(payload)
    except Exception as exc:  # noqa: BLE001 - the receipt records exact parse/validation blockers.
        receipt["blocked_reason"] = f"upstream_gate_invalid:{type(exc).__name__}"
        return receipt

    correction_spec = payload.get("correction_spec", {})
    if not isinstance(correction_spec, Mapping):
        correction_spec = {}
    receipt.update(
        {
            "schema": payload.get("schema"),
            "experiment": payload.get("experiment"),
            "run_date": payload.get("run_date"),
            "kernel_audit_ready_score": payload.get("kernel_audit_ready_score"),
            "final_kernel": correction_spec.get("final_kernel"),
            "correction_applied": payload.get("correction_applied"),
            "broken_kernel_controls_rejected": payload.get("broken_kernel_controls_rejected"),
            "large_n_timing_tuned": correction_spec.get("large_n_timing_tuned"),
            "quality_gate_specification": payload.get("quality_gate_specification", []),
            "source_reproducibility_checksum": payload.get("reproducibility_checksum"),
        }
    )
    ready = (
        payload.get("experiment") == exp5622.EXPERIMENT
        and payload.get("run_date") == RUN_DATE
        and payload.get("kernel_audit_ready_score") == 1.0
        and payload.get("correction_applied") is True
        and payload.get("broken_kernel_controls_rejected") is True
        and correction_spec.get("final_kernel") == CORRECTED_CDLS_METHOD
        and correction_spec.get("large_n_timing_tuned") is False
        and float(correction_spec.get("proposal_std", -1.0)) == exp5622.CDLS_PROPOSAL_STD
        and float(correction_spec.get("drift_scale", -1.0)) == exp5622.CDLS_DRIFT_SCALE
    )
    receipt["ready"] = bool(ready)
    receipt["blocked_reason"] = None if ready else "upstream_gate_not_ready"
    return receipt


def build_exact_ising_instances(instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES) -> list[IsingInstance]:
    """Create deterministic sparse Ising Hamiltonians for the crossover range."""

    return [_build_one_exact_instance(int(size)) for size in instance_sizes]


def _build_one_exact_instance(size: int) -> IsingInstance:
    if size <= 0:
        raise ValueError("instance size must be positive")
    target = np.array([1.0 if ((index * 37 + size) % 11) < 6 else -1.0 for index in range(size)], dtype=np.float32)
    biases = np.array(
        [0.09 * target[index] + 0.015 * ((index % 5) - 2) for index in range(size)],
        dtype=np.float32,
    )
    couplings = np.zeros((size, size), dtype=np.float32)
    for index in range(size):
        _add_coupling(couplings, index, (index + 1) % size, 0.13 * target[index] * target[(index + 1) % size])
        stride = 17 if size > 64 else 5
        peer = (index + stride) % size
        if peer != index:
            _add_coupling(couplings, index, peer, 0.045 * target[index] * target[peer])
        if index % 7 == 0:
            peer = (index + max(3, size // 8)) % size
            if peer != index:
                _add_coupling(couplings, index, peer, -0.025 * target[index] * target[peer])
    constraint_indices = tuple(range(0, size, max(1, size // 64)))
    checksum = sha256_json(
        {
            "generator": "exp5623_deterministic_sparse_ising_v1",
            "size": size,
            "biases": np.round(biases, 8).tolist(),
            "edges": _edge_list(couplings),
            "target": target.astype(int).tolist(),
            "constraint_indices": list(constraint_indices),
        }
    )
    return IsingInstance(
        instance_id=f"exact_sparse_ising_n{size}_{checksum[:10]}",
        size=size,
        descriptor_ids=("exp5623_deterministic_sparse_ising_v1",),
        couplings=couplings,
        biases=biases,
        target_spins=target,
        constraint_indices=constraint_indices,
        checksum=checksum,
    )


def _add_coupling(couplings: np.ndarray, i: int, j: int, value: float) -> None:
    couplings[i, j] = np.float32(couplings[i, j] + value)
    couplings[j, i] = np.float32(couplings[j, i] + value)


def _edge_list(couplings: np.ndarray) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    upper = np.triu(couplings, k=1)
    for i, j in np.argwhere(np.abs(upper) > 0.0):
        rows.append([int(i), int(j), round(float(couplings[int(i), int(j)]), 8)])
    return rows


def target_descriptors_for_instances(instances: Sequence[IsingInstance]) -> list[JsonDict]:
    """Describe targets with enough Hamiltonian detail for independent replay."""

    descriptors: list[JsonDict] = []
    for instance in instances:
        edges = _edge_list(instance.couplings)
        descriptors.append(
            {
                "instance_id": instance.instance_id,
                "target_family": "deterministic_exact_sparse_ising",
                "generator": "exp5623_deterministic_sparse_ising_v1",
                "size": int(instance.size),
                "temperature": DEFAULT_TEMPERATURE,
                "field_values": np.round(instance.biases, 8).tolist(),
                "edge_list": edges,
                "edge_count": len(edges),
                "constraint_indices": list(instance.constraint_indices),
                "target_spins": instance.target_spins.astype(int).tolist(),
                "hamiltonian": "E(x) = -0.5 x^T J x - h^T x with x in {-1,+1}^n",
                "descriptor_checksum": instance.checksum,
                "couplings_checksum": sha256_json(edges),
                "biases_checksum": sha256_json(np.round(instance.biases, 8).tolist()),
                "target_spins_checksum": sha256_json(instance.target_spins.astype(int).tolist()),
            }
        )
    return descriptors


def matched_schedule_for_instances(instances: Sequence[IsingInstance]) -> JsonDict:
    """Build the shared schedule for both samplers and both devices."""

    params = corrected_cdls_kernel_parameters()
    return {
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "retained_samples": DEFAULT_SAMPLES_PER_PAIR,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "stopping_rule": DEFAULT_STOPPING_RULE,
        "measurement_boundaries_shared": True,
        "target_hamiltonian_shared": True,
        "cdls_continuous_bound": params["continuous_bound"],
        "cdls_proposal_std": params["proposal_std"],
        "cdls_drift_scale": params["drift_scale"],
        "corrected_kernel": params["final_kernel"],
        "large_n_timing_tuned": False,
        "instance_checksums": {instance.instance_id: instance.checksum for instance in instances},
    }


def parse_meminfo(text: str) -> JsonDict:
    """Parse Linux memory totals while tolerating non-Linux hosts."""

    result: JsonDict = {}
    for source, target in (("MemTotal", "mem_total_kib"), ("MemAvailable", "mem_available_kib")):
        match = re.search(rf"^{source}:\s+(\d+)", text, flags=re.MULTILINE)
        if match:
            result[target] = int(match.group(1))
    return result


def _read_meminfo() -> str:
    path = Path("/proc/meminfo")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def cpu_device_receipt(tensor_runtime: Any | None = None) -> JsonDict:
    """Collect CPU identity and runtime metadata without implying a board claim."""

    return {
        "status": "reachable",
        "device_identities": [_cpu_model_name()],
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensor_runtime": str(getattr(tensor_runtime, "__version__", "unavailable")),
        },
        "driver_versions": {},
        "memory": parse_meminfo(_read_meminfo()),
        "metadata": {"python_executable": sys.executable, "platform": platform.platform()},
    }


def cuda_device_receipt(tensor_runtime: Any | None = None) -> JsonDict:
    """Collect CUDA identity, driver/runtime, and memory, or a precise blocker."""

    if tensor_runtime is None:
        try:
            tensor_runtime = _import_tensor_runtime()
        except Exception as exc:  # noqa: BLE001 - import failures are precondition evidence.
            return _blocked_cuda_receipt(f"tensor_runtime_import_failed:{type(exc).__name__}")
    cuda = getattr(tensor_runtime, "cuda", None)
    runtime_versions = {
        "tensor_runtime": str(getattr(tensor_runtime, "__version__", "unknown")),
        "cuda_runtime": str(getattr(getattr(tensor_runtime, "version", None), "cuda", "unknown")),
    }
    if cuda is None or not bool(cuda.is_available()):
        receipt = _blocked_cuda_receipt("cuda_unavailable")
        receipt["runtime_versions"] = runtime_versions
        return receipt

    count = int(cuda.device_count())
    identities = [str(cuda.get_device_name(index)) for index in range(count)]
    memory_rows: list[JsonDict] = []
    for index in range(count):
        row: JsonDict = {"index": index}
        try:
            free_bytes, total_bytes = cuda.mem_get_info(index)
            row["free_mib"] = int(free_bytes) // (1024 * 1024)
            row["total_mib"] = int(total_bytes) // (1024 * 1024)
        except Exception as exc:  # noqa: BLE001 - memory APIs vary by runtime version.
            row["memory_blocker"] = type(exc).__name__
        try:
            row["reserved_mib"] = int(cuda.memory_reserved(index)) // (1024 * 1024)
        except Exception:
            row.setdefault("reserved_mib", None)
        memory_rows.append(row)

    accelerator_cli = _query_accelerator_cli()
    return {
        "status": "reachable",
        "device_identities": identities,
        "runtime_versions": runtime_versions,
        "driver_versions": accelerator_cli.get("driver_versions", {}),
        "memory": {"device_memory": memory_rows, **accelerator_cli.get("memory", {})},
        "metadata": {"device_count": count},
        "blocked_reason": None,
    }


def _blocked_cuda_receipt(reason: str) -> JsonDict:
    return {
        "status": "blocked",
        "device_identities": [],
        "runtime_versions": {"tensor_runtime": "unavailable", "cuda_runtime": "unavailable"},
        "driver_versions": {},
        "memory": {},
        "metadata": {},
        "blocked_reason": reason,
    }


def _import_tensor_runtime() -> Any:  # pragma: no cover - live environment import.
    import torch

    return torch


def _query_accelerator_cli() -> JsonDict:  # pragma: no cover - depends on host drivers.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return {"driver_versions": {}, "memory": {}}
    if result.returncode != 0:
        return {"driver_versions": {}, "memory": {}}
    drivers: list[str] = []
    rows: list[JsonDict] = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            drivers.append(parts[1])
            rows.append(
                {
                    "index": index,
                    "name": parts[0],
                    "total_mib": _safe_int(parts[2]),
                    "free_mib": _safe_int(parts[3]),
                }
            )
    return {
        "driver_versions": {"nvidia_driver": drivers[0]} if drivers else {},
        "memory": {"accelerator_cli": rows} if rows else {},
    }


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def estimate_device_memory_mib(size: int) -> int:
    """Estimate row memory for dense tensors plus traces with a safety margin."""

    dense_bytes = int(size) * int(size) * 4
    vector_bytes = int(size) * 4
    return max(64, int((dense_bytes * 4 + vector_bytes * 16) / (1024 * 1024)) + 64)


def memory_permits_size(cuda_receipt: Mapping[str, Any], size: int) -> bool:
    """Return whether CUDA free memory appears sufficient for this size."""

    if cuda_receipt.get("status") != "reachable":
        return False
    required = estimate_device_memory_mib(size)
    memory = cuda_receipt.get("memory", {})
    if not isinstance(memory, Mapping):
        return False
    rows = memory.get("device_memory", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not rows:
        return True
    free_values = [row.get("free_mib") for row in rows if isinstance(row, Mapping) and row.get("free_mib") is not None]
    return bool(free_values) and max(int(value) for value in free_values) >= required


def memory_preflight_by_size(cuda_receipt: Mapping[str, Any], instance_sizes: Sequence[int]) -> list[JsonDict]:
    """Record which requested sizes are allowed to attempt CUDA timing."""

    return [
        {
            "size": int(size),
            "required_mib_estimate": estimate_device_memory_mib(int(size)),
            "cuda_memory_permits": memory_permits_size(cuda_receipt, int(size)),
        }
        for size in instance_sizes
    ]


def run_matched_sampler_rows(
    instances: list[IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: JsonDict,
    tensor_runtime: Any,
    clock: Clock = time.perf_counter,
    row_timeout_s: float | None = DEFAULT_ROW_TIMEOUT_S,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - exercised only on live tensor runtimes.
    """Run both methods on CPU/CUDA, preserving failures as rows."""

    rows: list[JsonDict] = []
    stats: list[JsonDict] = []
    for instance in instances:
        for seed in seeds:
            for device in DEVICES:
                for method_id in METHOD_IDS:
                    try:
                        if method_id == DISCRETE_METHOD:
                            row, stat = run_discrete_baseline_row(
                                instance=instance,
                                device=device,
                                seed=seed,
                                samples_per_pair=samples_per_pair,
                                matched_schedule=matched_schedule,
                                tensor_runtime=tensor_runtime,
                                clock=clock,
                                row_timeout_s=row_timeout_s,
                            )
                        else:
                            row, stat = run_corrected_cdls_row(
                                instance=instance,
                                device=device,
                                seed=seed,
                                samples_per_pair=samples_per_pair,
                                matched_schedule=matched_schedule,
                                tensor_runtime=tensor_runtime,
                                clock=clock,
                                row_timeout_s=row_timeout_s,
                            )
                    except TimeoutError:
                        row = blocked_timing_row(instance, seed, device, method_id, "timeout")
                        stat = blocked_sufficient_stat(row)
                    except RuntimeError as exc:
                        reason = "oom" if _is_oom_error(exc) else f"runtime_error:{type(exc).__name__}"
                        row = blocked_timing_row(instance, seed, device, method_id, reason)
                        stat = blocked_sufficient_stat(row)
                    except Exception as exc:  # noqa: BLE001 - failed rows are evidence, not reasons to drop a pair.
                        row = blocked_timing_row(instance, seed, device, method_id, f"failed:{type(exc).__name__}")
                        stat = blocked_sufficient_stat(row)
                    rows.append(row)
                    stats.append(stat)
    return rows, stats


def run_discrete_baseline_row(
    *,
    instance: IsingInstance,
    device: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    tensor_runtime: Any,
    clock: Clock,
    row_timeout_s: float | None,
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - live tensor runtime path.
    """Run the Exp5573 heat-bath transition rule while retaining traces."""

    return _run_tensor_sampler_row(
        instance=instance,
        device=device,
        seed=seed,
        samples_per_pair=samples_per_pair,
        matched_schedule=matched_schedule,
        tensor_runtime=tensor_runtime,
        clock=clock,
        row_timeout_s=row_timeout_s,
        method_id=DISCRETE_METHOD,
    )


def run_corrected_cdls_row(
    *,
    instance: IsingInstance,
    device: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    tensor_runtime: Any,
    clock: Clock,
    row_timeout_s: float | None,
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - live tensor runtime path.
    """Run the Exp5622 corrected projected cDLS kernel while retaining traces."""

    return _run_tensor_sampler_row(
        instance=instance,
        device=device,
        seed=seed,
        samples_per_pair=samples_per_pair,
        matched_schedule=matched_schedule,
        tensor_runtime=tensor_runtime,
        clock=clock,
        row_timeout_s=row_timeout_s,
        method_id=CORRECTED_CDLS_METHOD,
    )


def _run_tensor_sampler_row(
    *,
    instance: IsingInstance,
    device: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    tensor_runtime: Any,
    clock: Clock,
    row_timeout_s: float | None,
    method_id: str,
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - live tensor runtime path.
    runtime = tensor_runtime
    target_device = runtime.device("cuda:0" if device == "cuda" else "cpu")
    dtype = runtime.float32
    generator = runtime.Generator(device=target_device)
    generator.manual_seed(int(seed))
    started = clock()
    setup_start = clock()
    edge_array = np.asarray(_edge_list(instance.couplings), dtype=np.float32)
    edge_i = runtime.tensor(edge_array[:, 0].astype(np.int64), device=target_device, dtype=runtime.long)
    edge_j = runtime.tensor(edge_array[:, 1].astype(np.int64), device=target_device, dtype=runtime.long)
    edge_w = runtime.tensor(edge_array[:, 2], device=target_device, dtype=dtype)
    biases = runtime.tensor(instance.biases, device=target_device, dtype=dtype)
    target = runtime.tensor(instance.target_spins, device=target_device, dtype=dtype)
    constraint_indices = runtime.tensor(instance.constraint_indices, device=target_device, dtype=runtime.long)
    beta = 1.0 / float(matched_schedule["temperature"])
    spins = runtime.where(
        runtime.rand(instance.size, device=target_device, generator=generator) < 0.5,
        runtime.tensor(-1.0, device=target_device, dtype=dtype),
        runtime.tensor(1.0, device=target_device, dtype=dtype),
    )
    _sync_if_cuda(runtime, device)
    compile_time_s = max(clock() - setup_start, 0.0)
    memory_before = memory_snapshot(runtime, device)

    accepted = 0
    rejected = 0
    warmup_start = clock()
    for _ in range(int(matched_schedule["warmup_steps"])):
        _raise_if_timeout(clock, started, row_timeout_s)
        if method_id == DISCRETE_METHOD:
            spins = _sparse_heat_bath_step(runtime, spins, edge_i, edge_j, edge_w, biases, beta, generator)
            accepted += 1
        else:
            spins, step_accepted = _corrected_cdls_step(
                runtime,
                spins,
                edge_i,
                edge_j,
                edge_w,
                biases,
                beta,
                generator,
                float(matched_schedule["cdls_continuous_bound"]),
                float(matched_schedule["cdls_proposal_std"]),
                float(matched_schedule["cdls_drift_scale"]),
            )
            accepted += int(step_accepted)
            rejected += int(not step_accepted)
    _sync_if_cuda(runtime, device)
    warmup_time_s = max(clock() - warmup_start, 0.0)

    energies: list[float] = []
    constraints: list[float] = []
    sample_start = clock()
    thinning = int(matched_schedule["thinning"])
    total_steps = int(samples_per_pair) * thinning
    for step in range(total_steps):
        _raise_if_timeout(clock, started, row_timeout_s)
        if method_id == DISCRETE_METHOD:
            spins = _sparse_heat_bath_step(runtime, spins, edge_i, edge_j, edge_w, biases, beta, generator)
            accepted += 1
        else:
            spins, step_accepted = _corrected_cdls_step(
                runtime,
                spins,
                edge_i,
                edge_j,
                edge_w,
                biases,
                beta,
                generator,
                float(matched_schedule["cdls_continuous_bound"]),
                float(matched_schedule["cdls_proposal_std"]),
                float(matched_schedule["cdls_drift_scale"]),
            )
            accepted += int(step_accepted)
            rejected += int(not step_accepted)
        if (step + 1) % thinning == 0:
            energy = _sparse_ising_energy(runtime, spins, edge_i, edge_j, edge_w, biases)
            energies.append(float(energy.detach().cpu().item()))
            satisfied = (spins[constraint_indices] == target[constraint_indices]).to(dtype).mean()
            constraints.append(float(satisfied.detach().cpu().item()))
    _sync_if_cuda(runtime, device)
    sample_time_s = max(clock() - sample_start, 0.0)
    end_to_end = max(clock() - started, 0.0)
    proposals = accepted + rejected
    acceptance_rate = 1.0 if method_id == DISCRETE_METHOD else (accepted / proposals if proposals else 0.0)
    metrics = metrics_from_traces(energies, constraints, acceptance_rate=acceptance_rate)
    rid = row_id(instance.instance_id, seed, device, method_id)
    row: JsonDict = {
        "status": "success",
        "row_id": rid,
        "pair_id": f"{instance.instance_id}:seed{seed}",
        "method_id": method_id,
        "device": device,
        "backend": device,
        "instance_id": instance.instance_id,
        "size": int(instance.size),
        "seed": int(seed),
        "samples": int(samples_per_pair),
        "temperature": float(matched_schedule["temperature"]),
        "warmup_steps": int(matched_schedule["warmup_steps"]),
        "thinning": thinning,
        "precision": str(matched_schedule["precision"]),
        "acceptance_rate": round(float(acceptance_rate), 8),
        "best_energy": metrics["best_energy"],
        "energy_mean": metrics["mean_energy"],
        "energy_std": metrics["energy_std"],
        "energy_min": metrics["energy_min"],
        "energy_max": metrics["energy_max"],
        "energy_quantiles": metrics["energy_quantiles"],
        "exact_constraint_satisfaction_rate": metrics["exact_constraint_satisfaction_rate"],
        "autocorrelation_time": metrics["integrated_autocorrelation_time"],
        "effective_sample_size": metrics["effective_sample_size"],
        "compile_time_s": round(float(compile_time_s), 8),
        "warmup_time_s": round(float(warmup_time_s), 8),
        "sample_time_s": round(float(sample_time_s), 8),
        "wall_time_s": round(float(warmup_time_s + sample_time_s), 8),
        "end_to_end_wall_time_s": round(float(end_to_end), 8),
        "memory_before": memory_before,
        "memory_after": memory_snapshot(runtime, device),
        "kernel_device_path": f"tensor_{device}_{method_id}",
        "result_hash": sha256_json({"row_id": rid, "energies": [round(v, 6) for v in energies]}),
    }
    stat = sufficient_stat_from_trace(row, energies, constraints)
    return row, stat


def _corrected_cdls_step(
    runtime: Any,
    spins: Any,
    edge_i: Any,
    edge_j: Any,
    edge_w: Any,
    biases: Any,
    beta: float,
    generator: Any,
    continuous_bound: float,
    proposal_std: float,
    drift_scale: float,
) -> tuple[Any, bool]:  # pragma: no cover - live tensor runtime path.
    field = _sparse_field(runtime, spins, edge_i, edge_j, edge_w, biases)
    mean = spins + float(drift_scale) * float(beta) * field
    noise = runtime.randn(spins.shape, device=spins.device, generator=generator, dtype=spins.dtype)
    continuous = runtime.clamp(mean + float(proposal_std) * noise, -continuous_bound, continuous_bound)
    proposed = runtime.where(continuous >= 0.0, runtime.ones_like(spins), -runtime.ones_like(spins))
    current_energy = _sparse_ising_energy(runtime, spins, edge_i, edge_j, edge_w, biases)
    proposed_energy = _sparse_ising_energy(runtime, proposed, edge_i, edge_j, edge_w, biases)
    log_forward = _log_projected_proposal_probability(
        runtime,
        projected=proposed,
        source=spins,
        edge_i=edge_i,
        edge_j=edge_j,
        edge_w=edge_w,
        biases=biases,
        beta=beta,
        proposal_std=proposal_std,
        drift_scale=drift_scale,
    )
    log_reverse = _log_projected_proposal_probability(
        runtime,
        projected=spins,
        source=proposed,
        edge_i=edge_i,
        edge_j=edge_j,
        edge_w=edge_w,
        biases=biases,
        beta=beta,
        proposal_std=proposal_std,
        drift_scale=drift_scale,
    )
    log_accept = -float(beta) * (proposed_energy - current_energy) + log_reverse - log_forward
    uniform = runtime.rand((), device=spins.device, generator=generator, dtype=spins.dtype)
    accepted = bool((runtime.log(uniform.clamp_min(1e-12)) < runtime.minimum(log_accept, runtime.zeros_like(log_accept))).item())
    return (proposed if accepted else spins), accepted


def _log_projected_proposal_probability(
    runtime: Any,
    *,
    projected: Any,
    source: Any,
    edge_i: Any,
    edge_j: Any,
    edge_w: Any,
    biases: Any,
    beta: float,
    proposal_std: float,
    drift_scale: float,
) -> Any:  # pragma: no cover - live tensor runtime path.
    field = _sparse_field(runtime, source, edge_i, edge_j, edge_w, biases)
    mean = source + float(drift_scale) * float(beta) * field
    z = projected * mean / (float(proposal_std) * sqrt(2.0))
    probabilities = 0.5 * runtime.erfc(-z)
    return runtime.log(probabilities.clamp_min(1e-12)).sum()


def _sparse_field(runtime: Any, spins: Any, edge_i: Any, edge_j: Any, edge_w: Any, biases: Any) -> Any:  # pragma: no cover - live tensor runtime path.
    field = biases.clone()
    field.index_add_(0, edge_i, edge_w * spins[edge_j])
    field.index_add_(0, edge_j, edge_w * spins[edge_i])
    return field


def _sparse_heat_bath_step(
    runtime: Any,
    spins: Any,
    edge_i: Any,
    edge_j: Any,
    edge_w: Any,
    biases: Any,
    beta: float,
    generator: Any,
) -> Any:  # pragma: no cover - live tensor runtime path.
    field = _sparse_field(runtime, spins, edge_i, edge_j, edge_w, biases)
    probs = runtime.sigmoid(2.0 * beta * field)
    draws = runtime.rand(spins.shape, device=spins.device, generator=generator)
    return runtime.where(draws < probs, runtime.ones_like(spins), -runtime.ones_like(spins))


def _sparse_ising_energy(runtime: Any, spins: Any, edge_i: Any, edge_j: Any, edge_w: Any, biases: Any) -> Any:  # pragma: no cover - live tensor runtime path.
    pair_term = -runtime.sum(edge_w * spins[edge_i] * spins[edge_j])
    field_term = -runtime.dot(biases, spins)
    return pair_term + field_term


def _sync_if_cuda(runtime: Any, device: str) -> None:  # pragma: no cover - live tensor runtime path.
    if device == "cuda":
        cuda = getattr(runtime, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.synchronize()


def _raise_if_timeout(clock: Clock, started: float, row_timeout_s: float | None) -> None:
    if row_timeout_s is not None and max(clock() - started, 0.0) > float(row_timeout_s):
        raise TimeoutError("row timeout")


def _is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def memory_snapshot(runtime: Any, device: str) -> JsonDict:  # pragma: no cover - live tensor runtime path.
    if device == "cuda":
        cuda = getattr(runtime, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return {"status": "blocked", "blocked_reason": "cuda_unavailable"}
        try:
            free_bytes, total_bytes = cuda.mem_get_info(0)
            return {
                "status": "reachable",
                "free_mib": int(free_bytes) // (1024 * 1024),
                "total_mib": int(total_bytes) // (1024 * 1024),
            }
        except Exception as exc:  # noqa: BLE001 - memory hooks can be missing.
            return {"status": "blocked", "blocked_reason": type(exc).__name__}
    return {"status": "reachable", **parse_meminfo(_read_meminfo())}


def metrics_from_traces(
    energy_trace: Sequence[float],
    constraint_trace: Sequence[float],
    *,
    acceptance_rate: float,
) -> JsonDict:
    """Compute energy, constraint, ESS, and autocorrelation metrics from traces."""

    energies = np.asarray(energy_trace, dtype=np.float64)
    constraints = np.asarray(constraint_trace, dtype=np.float64)
    if energies.size == 0:
        raise ValueError("energy_trace must be nonempty")
    tau = integrated_autocorrelation_time(energies)
    ess = float(energies.size / tau) if tau > 0.0 else float(energies.size)
    return {
        "samples": int(energies.size),
        "acceptance_rate": round(float(acceptance_rate), 8),
        "best_energy": round(float(np.min(energies)), 8),
        "mean_energy": round(float(np.mean(energies)), 8),
        "energy_std": round(float(np.std(energies)), 8),
        "energy_min": round(float(np.min(energies)), 8),
        "energy_max": round(float(np.max(energies)), 8),
        "energy_quantiles": {
            "p05": round(float(np.quantile(energies, 0.05)), 8),
            "p50": round(float(np.quantile(energies, 0.50)), 8),
            "p95": round(float(np.quantile(energies, 0.95)), 8),
        },
        "exact_constraint_satisfaction_rate": round(float(np.mean(constraints)) if constraints.size else 0.0, 8),
        "integrated_autocorrelation_time": round(float(tau), 8),
        "effective_sample_size": round(float(ess), 8),
    }


def integrated_autocorrelation_time(values: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate integrated autocorrelation time using the initial-positive sum."""

    if values.size < 2:
        return 1.0
    centered = values.astype(np.float64) - float(np.mean(values))
    variance = float(np.dot(centered, centered) / centered.size)
    if variance <= 1e-18:
        return 1.0
    lag_limit = min(values.size // 2, max_lag or 1000)
    positive_sum = 0.0
    for lag in range(1, lag_limit + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / ((values.size - lag) * variance))
        if corr <= 0.0:
            break
        positive_sum += corr
    return max(1.0, 1.0 + 2.0 * positive_sum)


def sufficient_stat_from_trace(
    row: Mapping[str, Any],
    energy_trace: Sequence[float],
    constraint_trace: Sequence[float],
) -> JsonDict:
    """Build the recomputable per-row evidence record."""

    energy_values = [round(float(value), 8) for value in energy_trace]
    constraint_values = [round(float(value), 8) for value in constraint_trace]
    return {
        "status": row.get("status", "success"),
        "row_id": row["row_id"],
        "pair_id": row["pair_id"],
        "method_id": row["method_id"],
        "device": row["device"],
        "instance_id": row["instance_id"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "samples": int(row.get("samples", len(energy_values))),
        "energy_trace": energy_values,
        "constraint_trace": constraint_values,
        "energy_trace_sha256": sha256_json(energy_values),
        "constraint_trace_sha256": sha256_json(constraint_values),
        "timing": {
            "compile_time_s": row.get("compile_time_s"),
            "warmup_time_s": row.get("warmup_time_s"),
            "sample_time_s": row.get("sample_time_s"),
            "end_to_end_wall_time_s": row.get("end_to_end_wall_time_s"),
        },
    }


def blocked_timing_row(instance: IsingInstance, seed: int, device: str, method_id: str, reason: str) -> JsonDict:
    """Create an explicit failed row that cannot enter speedups."""

    rid = row_id(instance.instance_id, seed, device, method_id)
    return {
        "status": "blocked",
        "blocked_reason": reason,
        "row_id": rid,
        "pair_id": f"{instance.instance_id}:seed{seed}",
        "method_id": method_id,
        "device": device,
        "backend": device,
        "instance_id": instance.instance_id,
        "size": int(instance.size),
        "seed": int(seed),
        "samples": 0,
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "acceptance_rate": None,
        "best_energy": None,
        "energy_mean": None,
        "energy_std": None,
        "energy_min": None,
        "energy_max": None,
        "energy_quantiles": None,
        "exact_constraint_satisfaction_rate": None,
        "autocorrelation_time": None,
        "effective_sample_size": None,
        "compile_time_s": None,
        "warmup_time_s": None,
        "sample_time_s": None,
        "wall_time_s": None,
        "end_to_end_wall_time_s": None,
        "memory_before": None,
        "memory_after": None,
        "kernel_device_path": f"tensor_{device}_{method_id}",
        "result_hash": None,
    }


def blocked_sufficient_stat(row: Mapping[str, Any]) -> JsonDict:
    """Create a sufficient-stat placeholder for a blocked timing row."""

    return {
        "status": "blocked",
        "blocked_reason": row.get("blocked_reason"),
        "row_id": row["row_id"],
        "pair_id": row["pair_id"],
        "method_id": row["method_id"],
        "device": row["device"],
        "instance_id": row["instance_id"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "samples": 0,
        "energy_trace": [],
        "constraint_trace": [],
        "energy_trace_sha256": sha256_json([]),
        "constraint_trace_sha256": sha256_json([]),
        "timing": {
            "compile_time_s": None,
            "warmup_time_s": None,
            "sample_time_s": None,
            "end_to_end_wall_time_s": None,
        },
    }


def blocked_rows_for_instances(instances: Sequence[IsingInstance], seeds: Sequence[int], reason: str) -> tuple[list[JsonDict], list[JsonDict]]:
    """Emit complete method/device blocker rows when a structured gate fails."""

    rows: list[JsonDict] = []
    stats: list[JsonDict] = []
    for instance in instances:
        for seed in seeds:
            for device in DEVICES:
                for method_id in METHOD_IDS:
                    row = blocked_timing_row(instance, int(seed), device, method_id, reason)
                    rows.append(row)
                    stats.append(blocked_sufficient_stat(row))
    return rows, stats


def write_sufficient_statistics(root: str | Path, stat_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Write recomputable traces and return relative path plus checksum."""

    path = Path(root) / SUFFICIENT_STATISTICS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "carnot.experiment_5623.sufficient_statistics.v1",
        "experiment": EXPERIMENT,
        "row_count": len(stat_rows),
        "rows": list(stat_rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return SUFFICIENT_STATISTICS_RELATIVE_PATH.as_posix(), file_sha256(path)


def recompute_metrics_from_sufficient_statistics(path: str | Path) -> JsonDict:
    """Recompute row-level metrics from the saved trace evidence."""

    payload = read_json(Path(path))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("sufficient statistics rows must be a list")
    recomputed_rows: list[JsonDict] = []
    samples: list[int] = []
    pairs: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pairs.add(str(row.get("pair_id", "")))
        energy = row.get("energy_trace", [])
        constraint = row.get("constraint_trace", [])
        row_status = row.get("status", "success" if energy else "blocked")
        if row_status != "success":
            continue
        if not isinstance(energy, Sequence) or isinstance(energy, (str, bytes, bytearray)):
            continue
        if not isinstance(constraint, Sequence) or isinstance(constraint, (str, bytes, bytearray)):
            continue
        metrics = metrics_from_traces(energy, constraint, acceptance_rate=0.5)
        samples.append(int(metrics["samples"]))
        recomputed_rows.append(
            {
                "row_id": row.get("row_id"),
                "samples": metrics["samples"],
                "mean_energy": metrics["mean_energy"],
                "integrated_autocorrelation_time": metrics["integrated_autocorrelation_time"],
                "effective_sample_size": metrics["effective_sample_size"],
                "exact_constraint_satisfaction_rate": metrics["exact_constraint_satisfaction_rate"],
            }
        )
    return {
        "row_count": len(rows),
        "successful_row_count": len(recomputed_rows),
        "pair_count": len({pair for pair in pairs if pair}),
        "samples_min": min(samples) if samples else 0,
        "rows": recomputed_rows,
        "source_sha256": file_sha256(Path(path)),
    }


def summarize_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    stat_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], int | None, bool]:
    """Summarize quality and speed only after pair-level gates are known."""

    rows_by_pair = _group_rows_by_pair(raw_rows)
    stats_by_row = {str(row.get("row_id")): row for row in stat_rows if isinstance(row, Mapping)}
    quality_results: list[JsonDict] = []
    energy_metrics: list[JsonDict] = []
    mixing_metrics: list[JsonDict] = []
    successful_pairs: list[JsonDict] = []
    speedups: list[JsonDict] = []

    for pair_id, rows in sorted(rows_by_pair.items()):
        quality = quality_gate_result(pair_id, rows, stats_by_row)
        quality_results.append(quality)
        if quality["energy_quality"] is not None:
            energy_metrics.append(quality["energy_quality"])
        if quality["mixing_quality"] is not None:
            mixing_metrics.append(quality["mixing_quality"])
        if quality["included_in_speedups"]:
            successful_pairs.append(
                {
                    "pair_id": pair_id,
                    "instance_id": quality["instance_id"],
                    "size": quality["size"],
                    "seed": quality["seed"],
                    "row_ids": [rows[key]["row_id"] for key in _required_row_keys()],
                }
            )
            speedups.append(speedup_for_pair(pair_id, rows))

    intervals = timing_intervals_by_size(speedups)
    crossover_size, crossover_allowed = crossover_from_intervals(intervals)
    return energy_metrics, mixing_metrics, quality_results, successful_pairs, speedups, crossover_size, crossover_allowed


def _group_rows_by_pair(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    grouped: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for row in raw_rows:
        pair_id = str(row.get("pair_id", ""))
        method_id = str(row.get("method_id", ""))
        device = str(row.get("device", ""))
        grouped.setdefault(pair_id, {})[(method_id, device)] = row
    return grouped


def _required_row_keys() -> tuple[tuple[str, str], ...]:
    return (
        (DISCRETE_METHOD, "cpu"),
        (DISCRETE_METHOD, "cuda"),
        (CORRECTED_CDLS_METHOD, "cpu"),
        (CORRECTED_CDLS_METHOD, "cuda"),
    )


def quality_gate_result(
    pair_id: str,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    stats_by_row: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Explain whether one size/seed pair can enter speedup summaries."""

    missing = [key for key in _required_row_keys() if key not in rows]
    sample_row = next(iter(rows.values())) if rows else {}
    base: JsonDict = {
        "pair_id": pair_id,
        "instance_id": sample_row.get("instance_id"),
        "size": int(sample_row.get("size", 0) or 0),
        "seed": int(sample_row.get("seed", 0) or 0),
        "included_in_speedups": False,
        "exclusion_reason": None,
        "device_results": {},
        "energy_quality": None,
        "mixing_quality": None,
    }
    if missing:
        base["exclusion_reason"] = f"missing_rows:{missing}"
        return base
    failed = [rows[key] for key in _required_row_keys() if rows[key].get("status", "success") != "success"]
    if failed:
        reasons = sorted({str(row.get("blocked_reason", "failed")) for row in failed})
        base["exclusion_reason"] = f"failed_rows:{','.join(reasons)}"
        return base
    if not _all_rows_match([rows[key] for key in _required_row_keys()]):
        base["exclusion_reason"] = "schedule_or_target_mismatch"
        return base
    if any(int(rows[key].get("samples", 0)) < MIN_SAMPLES_PER_SUCCESS for key in _required_row_keys()):
        base["exclusion_reason"] = "samples_below_floor"
        return base

    device_results: dict[str, JsonDict] = {}
    energy_by_device: dict[str, JsonDict] = {}
    mixing_by_device: dict[str, JsonDict] = {}
    for device in DEVICES:
        baseline = rows[(DISCRETE_METHOD, device)]
        candidate = rows[(CORRECTED_CDLS_METHOD, device)]
        result = _quality_for_device(baseline, candidate, stats_by_row)
        device_results[device] = result
        energy_by_device[device] = result["energy"]
        mixing_by_device[device] = result["mixing"]

    included = all(result["passes"] for result in device_results.values())
    base.update(
        {
            "included_in_speedups": included,
            "exclusion_reason": "quality_matched" if included else "quality_gate_failed",
            "device_results": device_results,
            "energy_quality": {
                "pair_id": pair_id,
                "instance_id": sample_row.get("instance_id"),
                "size": int(sample_row["size"]),
                "seed": int(sample_row["seed"]),
                "by_device": energy_by_device,
            },
            "mixing_quality": {
                "pair_id": pair_id,
                "instance_id": sample_row.get("instance_id"),
                "size": int(sample_row["size"]),
                "seed": int(sample_row["seed"]),
                "by_device": mixing_by_device,
            },
        }
    )
    return base


def _all_rows_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    keys = ("instance_id", "size", "seed", "samples", "temperature", "warmup_steps", "thinning", "precision")
    first = rows[0]
    return all(row.get(key) == first.get(key) for row in rows for key in keys)


def _quality_for_device(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stats_by_row: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    baseline_stats = stats_by_row.get(str(baseline.get("row_id")), {})
    candidate_stats = stats_by_row.get(str(candidate.get("row_id")), {})
    baseline_energy = _trace_array(baseline_stats.get("energy_trace", []))
    candidate_energy = _trace_array(candidate_stats.get("energy_trace", []))
    histogram_tv = energy_histogram_tv(baseline_energy, candidate_energy)
    mean_delta = float(candidate["energy_mean"]) - float(baseline["energy_mean"])
    mean_tolerance = max(
        float(QUALITY_THRESHOLDS["mean_energy_worse_abs_max"]),
        float(QUALITY_THRESHOLDS["mean_energy_worse_rel_max"]) * abs(float(baseline["energy_mean"])),
    )
    mean_ci = mean_delta_interval(baseline_energy, candidate_energy)
    best_delta = float(candidate["best_energy"]) - float(baseline["best_energy"])
    constraint_drop = float(baseline["exact_constraint_satisfaction_rate"]) - float(candidate["exact_constraint_satisfaction_rate"])
    tau_ratio = float(candidate["autocorrelation_time"]) / max(float(baseline["autocorrelation_time"]), 1e-12)
    min_ess = min(float(baseline["effective_sample_size"]), float(candidate["effective_sample_size"]))
    acceptance_rate = float(candidate["acceptance_rate"])
    gates = {
        "acceptance_valid": 0.0 < acceptance_rate <= 1.0
        and acceptance_rate >= float(QUALITY_THRESHOLDS["min_corrected_cdls_acceptance_rate"]),
        "energy_distribution_tv": histogram_tv <= float(QUALITY_THRESHOLDS["energy_histogram_tv_delta_max"]),
        "mean_energy_noninferior": mean_ci[1] <= mean_tolerance,
        "best_energy_noninferior": best_delta <= float(QUALITY_THRESHOLDS["best_energy_worse_abs_max"]),
        "constraint_noninferior": constraint_drop <= float(QUALITY_THRESHOLDS["constraint_satisfaction_rate_drop_max"]),
        "ess_floor": min_ess >= float(QUALITY_THRESHOLDS["min_effective_sample_size"]),
        "autocorrelation_ratio": tau_ratio <= float(QUALITY_THRESHOLDS["max_integrated_autocorrelation_ratio"]),
    }
    return {
        "passes": all(gates.values()),
        "gate_results": gates,
        "energy": {
            "discrete_dls_heat_bath": _energy_summary(baseline),
            "corrected_cdls_projection_mh": _energy_summary(candidate),
            "energy_histogram_tv": round(float(histogram_tv), 8),
            "mean_energy_delta_cdls_minus_dls": round(float(mean_delta), 8),
            "mean_energy_delta_ci_95": [round(float(mean_ci[0]), 8), round(float(mean_ci[1]), 8)],
            "best_energy_delta_cdls_minus_dls": round(float(best_delta), 8),
            "constraint_satisfaction_drop": round(float(constraint_drop), 8),
        },
        "mixing": {
            "discrete_dls_heat_bath": _mixing_summary(baseline),
            "corrected_cdls_projection_mh": _mixing_summary(candidate),
            "integrated_autocorrelation_ratio": round(float(tau_ratio), 8),
            "min_effective_sample_size": round(float(min_ess), 8),
            "corrected_cdls_acceptance_rate": round(float(acceptance_rate), 8),
        },
    }


def _trace_array(value: Any) -> np.ndarray:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return np.asarray(value, dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def energy_histogram_tv(left: np.ndarray, right: np.ndarray, bins: int = 32) -> float:
    """Compare two energy distributions with shared histogram bins."""

    if left.size == 0 or right.size == 0:
        return 1.0
    low = float(min(np.min(left), np.min(right)))
    high = float(max(np.max(left), np.max(right)))
    if abs(high - low) <= 1e-12:
        return 0.0 if abs(float(np.mean(left)) - float(np.mean(right))) <= 1e-12 else 1.0
    left_counts, edges = np.histogram(left, bins=bins, range=(low, high))
    right_counts, _ = np.histogram(right, bins=edges)
    left_prob = left_counts.astype(np.float64) / float(np.sum(left_counts))
    right_prob = right_counts.astype(np.float64) / float(np.sum(right_counts))
    return float(0.5 * np.sum(np.abs(left_prob - right_prob)))


def mean_delta_interval(baseline_energy: np.ndarray, candidate_energy: np.ndarray) -> tuple[float, float]:
    """Return a conservative normal-approximation interval for mean energy delta."""

    if baseline_energy.size == 0 or candidate_energy.size == 0:
        return (float("inf"), float("inf"))
    delta = float(np.mean(candidate_energy) - np.mean(baseline_energy))
    variance = float(np.var(candidate_energy) / candidate_energy.size + np.var(baseline_energy) / baseline_energy.size)
    half_width = 1.96 * sqrt(max(variance, 0.0))
    return (delta - half_width, delta + half_width)


def _energy_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "best_energy": float(row["best_energy"]),
        "mean_energy": float(row["energy_mean"]),
        "energy_std": float(row["energy_std"]),
        "energy_min": float(row["energy_min"]),
        "energy_max": float(row["energy_max"]),
        "energy_quantiles": dict(row["energy_quantiles"]),
        "exact_constraint_satisfaction_rate": float(row["exact_constraint_satisfaction_rate"]),
        "acceptance_rate": float(row["acceptance_rate"]),
    }


def _mixing_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "integrated_autocorrelation_time": float(row["autocorrelation_time"]),
        "effective_sample_size": float(row["effective_sample_size"]),
    }


def speedup_for_pair(pair_id: str, rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> JsonDict:
    """Compute matched ratios for one included size/seed pair."""

    dls_cpu = rows[(DISCRETE_METHOD, "cpu")]
    dls_cuda = rows[(DISCRETE_METHOD, "cuda")]
    cdls_cpu = rows[(CORRECTED_CDLS_METHOD, "cpu")]
    cdls_cuda = rows[(CORRECTED_CDLS_METHOD, "cuda")]
    return {
        "pair_id": pair_id,
        "instance_id": dls_cpu["instance_id"],
        "size": int(dls_cpu["size"]),
        "seed": int(dls_cpu["seed"]),
        "discrete_dls_cuda_vs_cpu_speedup": _ratio(dls_cpu, dls_cuda),
        "corrected_cdls_cuda_vs_cpu_speedup": _ratio(cdls_cpu, cdls_cuda),
        "corrected_cdls_vs_discrete_dls_cpu_speedup": _ratio(dls_cpu, cdls_cpu),
        "corrected_cdls_vs_discrete_dls_cuda_speedup": _ratio(dls_cuda, cdls_cuda),
        "wall_time_s": {
            "cpu_discrete_dls_heat_bath": float(dls_cpu["wall_time_s"]),
            "cuda_discrete_dls_heat_bath": float(dls_cuda["wall_time_s"]),
            "cpu_corrected_cdls_projection_mh": float(cdls_cpu["wall_time_s"]),
            "cuda_corrected_cdls_projection_mh": float(cdls_cuda["wall_time_s"]),
        },
    }


def _ratio(numerator_row: Mapping[str, Any], denominator_row: Mapping[str, Any]) -> float | None:
    denominator = float(denominator_row["wall_time_s"])
    if denominator <= 0.0:
        return None
    return round(float(numerator_row["wall_time_s"]) / denominator, 8)


def timing_intervals_by_size(speedups: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate paired-seed speedup intervals by size."""

    by_size: dict[int, list[Mapping[str, Any]]] = {}
    for row in speedups:
        by_size.setdefault(int(row["size"]), []).append(row)
    intervals: list[JsonDict] = []
    for size, rows in sorted(by_size.items()):
        entry: JsonDict = {"size": size, "n_seed_pairs": len(rows)}
        for key in (
            "discrete_dls_cuda_vs_cpu_speedup",
            "corrected_cdls_cuda_vs_cpu_speedup",
            "corrected_cdls_vs_discrete_dls_cpu_speedup",
            "corrected_cdls_vs_discrete_dls_cuda_speedup",
        ):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            entry[f"{key}_interval_95"] = ratio_interval_95(values)
            entry[f"{key}_values"] = [round(float(value), 8) for value in values]
        interval = entry["corrected_cdls_cuda_vs_cpu_speedup_interval_95"]
        entry["excludes_1_favorable"] = bool(len(rows) >= MIN_PAIRED_SEEDS and interval[0] > 1.0)
        intervals.append(entry)
    return intervals


def ratio_interval_95(values: Sequence[float]) -> list[float]:
    """Return a log-ratio 95% interval across paired seeds."""

    if not values:
        return [float("nan"), float("nan")]
    positive = np.asarray([float(value) for value in values if float(value) > 0.0], dtype=np.float64)
    if positive.size == 0:
        return [float("nan"), float("nan")]
    logs = np.log(positive)
    if logs.size == 1:
        value = float(exp(float(logs[0])))
        return [round(value, 8), round(value, 8)]
    t_value = _student_t_975(int(logs.size - 1))
    half_width = t_value * float(np.std(logs, ddof=1)) / sqrt(float(logs.size))
    center = float(np.mean(logs))
    return [round(float(exp(center - half_width)), 8), round(float(exp(center + half_width)), 8)]


def _student_t_975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    return table.get(max(1, int(df)), 1.96)


def crossover_from_intervals(intervals: Sequence[Mapping[str, Any]]) -> tuple[int | None, bool]:
    """Return the smallest size with a favorable quality-matched timing interval."""

    for row in sorted(intervals, key=lambda item: int(item["size"])):
        if row.get("excludes_1_favorable") is True:
            return int(row["size"]), True
    return None, False


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    tensor_runtime: Any | None = None,
    sampler_runner: SamplerRunner = run_matched_sampler_rows,
    clock: Clock = time.perf_counter,
    instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
    row_timeout_s: float | None = DEFAULT_ROW_TIMEOUT_S,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5623 artifact and its sufficient-statistic sidecar."""

    started = clock()
    root_path = Path(root)
    upstream_receipt = load_upstream_gate_receipt(root_path)
    runtime_obj = tensor_runtime
    if runtime_obj is None:
        try:
            runtime_obj = _import_tensor_runtime()
        except Exception:
            runtime_obj = None
    cpu_receipt = cpu_device_receipt(runtime_obj)
    cuda_receipt = cuda_device_receipt(runtime_obj)
    instances = build_exact_ising_instances(instance_sizes)
    matched_schedule = matched_schedule_for_instances(instances)
    matched_schedule["retained_samples"] = int(samples_per_pair)
    memory_preflight = memory_preflight_by_size(cuda_receipt, instance_sizes)
    memory_by_size = {int(row["size"]): bool(row["cuda_memory_permits"]) for row in memory_preflight}
    runnable_instances = [instance for instance in instances if memory_by_size.get(instance.size, False)]
    blocked_memory_instances = [instance for instance in instances if not memory_by_size.get(instance.size, False)]

    preconditions: JsonDict = {
        "upstream_gate_ready": upstream_receipt["ready"],
        "cpu_available": cpu_receipt["status"] == "reachable",
        "cuda_available": cuda_receipt["status"] == "reachable",
        "memory_preflight_by_size": memory_preflight,
        "blocked_reasons": [],
    }
    if not upstream_receipt["ready"]:
        preconditions["blocked_reasons"].append(upstream_receipt["blocked_reason"])
    if cuda_receipt["status"] != "reachable":
        preconditions["blocked_reasons"].append(cuda_receipt.get("blocked_reason", "cuda_blocked"))

    raw_rows: list[JsonDict] = []
    stat_rows: list[JsonDict] = []
    if not preconditions["upstream_gate_ready"]:
        raw_rows, stat_rows = blocked_rows_for_instances(instances, seeds, str(upstream_receipt["blocked_reason"]))
    elif not preconditions["cuda_available"] or runtime_obj is None:
        reason = str(cuda_receipt.get("blocked_reason", "cuda_unavailable"))
        raw_rows, stat_rows = blocked_rows_for_instances(instances, seeds, reason)
    else:
        if blocked_memory_instances:
            blocked_rows, blocked_stats = blocked_rows_for_instances(
                blocked_memory_instances,
                seeds,
                "cuda_memory_preflight_blocked",
            )
            raw_rows.extend(blocked_rows)
            stat_rows.extend(blocked_stats)
        if runnable_instances:
            try:
                run_rows, run_stats = sampler_runner(
                    runnable_instances,
                    tuple(int(seed) for seed in seeds),
                    int(samples_per_pair),
                    matched_schedule,
                    runtime_obj,
                    clock,
                    row_timeout_s,
                )
                raw_rows.extend(run_rows)
                stat_rows.extend(run_stats)
            except Exception as exc:  # noqa: BLE001 - top-level failures become rows, not dropped evidence.
                preconditions["blocked_reasons"].append(f"sampler_run_failed:{type(exc).__name__}:{exc}")
                blocked_rows, blocked_stats = blocked_rows_for_instances(runnable_instances, seeds, f"sampler_run_failed:{type(exc).__name__}")
                raw_rows.extend(blocked_rows)
                stat_rows.extend(blocked_stats)

    sufficient_statistics_path, sufficient_statistics_sha256 = write_sufficient_statistics(root_path, stat_rows)
    energy_metrics, mixing_metrics, quality_results, successful_pairs, speedups, crossover_size, crossover_allowed = summarize_rows(
        raw_rows,
        stat_rows,
    )
    intervals = timing_intervals_by_size(speedups)
    crossover_size, crossover_allowed = crossover_from_intervals(intervals)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions": preconditions,
        "upstream_gate_receipt": upstream_receipt,
        "target_descriptors": target_descriptors_for_instances(instances),
        "instance_sizes": [int(size) for size in instance_sizes],
        "models_tested": models_tested(),
        "seeds": [int(seed) for seed in seeds],
        "random_seeds": [int(seed) for seed in seeds],
        "samples_per_pair": int(samples_per_pair),
        "matched_schedule": matched_schedule,
        "quality_gate_specification": quality_gate_specification(upstream_receipt),
        "cpu_device_receipt": cpu_receipt,
        "cuda_device_receipt": cuda_receipt,
        "timing_rows": [_timing_summary(row) for row in raw_rows],
        "energy_quality_metrics": energy_metrics,
        "mixing_metrics": mixing_metrics,
        "quality_gate_results_by_pair": quality_results,
        "successful_matched_pairs": successful_pairs,
        "speedup_by_pair": speedups,
        "timing_intervals_by_size": intervals,
        "sufficient_statistics_path": sufficient_statistics_path,
        "sufficient_statistics_sha256": sufficient_statistics_sha256,
        "crossover_size": crossover_size,
        "crossover_claim_allowed": crossover_allowed,
        "board_speedup_claimed": False,
        "tests_added_or_reused": _normalize_tests(tests_added_or_reused),
        "duration_s": round(max(clock() - started, 0.0), 8),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def quality_gate_specification(upstream_receipt: Mapping[str, Any]) -> JsonDict:
    """Return the downstream thresholds and the Exp5622 gate source."""

    return {
        "defined_before_timing": True,
        "source": "Exp5622 quality_gate_specification plus required acceptance and uncertainty checks",
        "upstream_quality_gate_count": len(upstream_receipt.get("quality_gate_specification", [])),
        "thresholds": dict(QUALITY_THRESHOLDS),
        "uncertainty_method": "mean_energy_delta_normal_interval_and_log_speedup_t_interval_across_paired_seeds",
    }


def _timing_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "status": row.get("status", "success"),
        "blocked_reason": row.get("blocked_reason"),
        "row_id": row["row_id"],
        "pair_id": row["pair_id"],
        "method_id": row["method_id"],
        "device": row["device"],
        "backend": row.get("backend", row["device"]),
        "instance_id": row["instance_id"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "samples": int(row.get("samples", 0)),
        "temperature": row.get("temperature"),
        "warmup_steps": row.get("warmup_steps"),
        "thinning": row.get("thinning"),
        "precision": row.get("precision"),
        "compile_time_s": row.get("compile_time_s"),
        "warmup_time_s": row.get("warmup_time_s"),
        "sample_time_s": row.get("sample_time_s"),
        "wall_time_s": row.get("wall_time_s"),
        "end_to_end_wall_time_s": row.get("end_to_end_wall_time_s"),
        "memory_before": row.get("memory_before"),
        "memory_after": row.get("memory_after"),
        "kernel_device_path": row.get("kernel_device_path"),
        "result_hash": row.get("result_hash"),
    }


def _normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    if tests_added_or_reused:
        return [str(item) for item in tests_added_or_reused]
    return [
        "tests/python/test_experiment_5623_cdls_multiseed_cpu_cuda_crossover.py",
        ".venv/bin/pytest tests/python -q",
    ]


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict without implying board or TSU evidence."""

    if payload.get("crossover_claim_allowed") is True:
        return (
            f"complete: quality-matched corrected-cDLS CPU/CUDA crossover at n={payload.get('crossover_size')}; "
            "board_speedup_claimed=false; no board or TSU result claimed"
        )
    blockers = list(payload.get("preconditions", {}).get("blocked_reasons", []))
    if payload.get("upstream_gate_receipt", {}).get("ready") is not True:
        return "blocked: Exp5622 corrected-kernel gate unavailable; no GPU work allocated"
    if payload.get("cuda_device_receipt", {}).get("status") != "reachable":
        return "blocked: CUDA unavailable; no crossover evidence and board_speedup_claimed=false"
    if len(payload.get("successful_matched_pairs", [])) == 0:
        return (
            "complete: no quality-matched crossover pairs entered speedups; "
            f"blocked_reasons={len(blockers)}; no-crossover evidence is terminal for this run"
        )
    return (
        "complete: quality-matched pairs recorded but timing interval did not prove crossover; "
        "no-crossover evidence is terminal for this run"
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate required fields and fail closed on unsupported claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("board_speedup_claimed") is not False:
        raise ValueError("board_speedup_claimed must be false")
    if int(payload.get("samples_per_pair", 0)) < MIN_SAMPLES_PER_SUCCESS:
        raise ValueError("samples_per_pair must be at least 10000")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < MIN_PAIRED_SEEDS:
        raise ValueError("seeds must contain at least five paired values")
    if payload.get("random_seeds") != seeds:
        raise ValueError("random_seeds must match seeds")
    model_ids = {row.get("model_id") for row in payload.get("models_tested", []) if isinstance(row, Mapping)}
    if model_ids != set(METHOD_IDS):
        raise ValueError("models_tested mismatch")
    for row in payload.get("models_tested", []):
        if not isinstance(row, Mapping):
            raise ValueError("models_tested rows must be mappings")
        if row.get("model_id") in FORBIDDEN_METHODS or row.get("biased_control_kernel_used") is not False:
            raise ValueError("biased_control_kernel_used must be false")
    stats_sha = str(payload.get("sufficient_statistics_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", stats_sha) or stats_sha == "0" * 64:
        raise ValueError("sufficient_statistics_sha256 invalid")
    successful_pairs = payload.get("successful_matched_pairs")
    speedups = payload.get("speedup_by_pair")
    if not isinstance(successful_pairs, list) or not isinstance(speedups, list):
        raise ValueError("successful_matched_pairs and speedup_by_pair must be lists")
    if len(successful_pairs) != len(speedups):
        raise ValueError("successful_matched_pairs must match speedup_by_pair length")
    success_ids = {row.get("pair_id") for row in successful_pairs if isinstance(row, Mapping)}
    if {row.get("pair_id") for row in speedups if isinstance(row, Mapping)} != success_ids:
        raise ValueError("speedup_by_pair must use only successful matched pairs")
    timing_rows = payload.get("timing_rows", [])
    if not isinstance(timing_rows, list):
        raise ValueError("timing_rows must be a list")
    for row in timing_rows:
        if not isinstance(row, Mapping):
            raise ValueError("timing_rows rows must be mappings")
        if row.get("method_id") in FORBIDDEN_METHODS:
            raise ValueError("timing_rows include forbidden biased control")
        if row.get("status") == "success" and int(row.get("samples", 0)) < MIN_SAMPLES_PER_SUCCESS:
            raise ValueError("timing_rows success below samples_per_pair floor")
    if payload.get("upstream_gate_receipt", {}).get("ready") is not True and successful_pairs:
        raise ValueError("upstream_gate_receipt not ready but successful pairs exist")
    expected_size, expected_allowed = crossover_from_intervals(payload.get("timing_intervals_by_size", []))
    if payload.get("crossover_claim_allowed") is not expected_allowed:
        raise ValueError("crossover_claim_allowed mismatch")
    if expected_allowed and payload.get("crossover_size") != expected_size:
        raise ValueError("crossover_size mismatch")
    if payload.get("crossover_claim_allowed") is True:
        matching = [
            row
            for row in payload.get("timing_intervals_by_size", [])
            if isinstance(row, Mapping) and row.get("size") == payload.get("crossover_size")
        ]
        if not matching or matching[0].get("excludes_1_favorable") is not True:
            raise ValueError("crossover_claim_allowed requires favorable timing interval")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal Exp5623 artifact with stable formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    tests_added_or_reused: Sequence[str] | None = None,
) -> Path:  # pragma: no cover - thin live runner.
    """Build, validate, and write Exp5623 outputs."""

    artifact = build_artifact(root=repo_root, tests_added_or_reused=tests_added_or_reused)
    return write_output(repo_root, artifact)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    print(run_experiment())
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
