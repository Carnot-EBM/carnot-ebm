"""Exp5611 bounded cDLS matched CPU/CUDA crossover benchmark.

Spec refs: REQ-SAMPLE-5611, SCENARIO-SAMPLE-5611.

This experiment treats continuous-intermediate discrete Langevin sampling as a
local hypothesis, not as a performance authority. The existing Exp5573
discrete heat-bath baseline is called unchanged. The new cDLS method proposes a
bounded continuous intermediate, projects it back to Ising spins, then applies a
Metropolis-Hastings correction using the exact discrete Ising energy and the
proposal probability for that projected spin state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_5573_matched_sampler_hardware_continuity as exp5573


JsonDict = dict[str, Any]
Clock = Callable[[], float]
SamplerRunner = Callable[
    [list[exp5573.IsingInstance], tuple[int, ...], int, JsonDict, Any, Clock],
    list[JsonDict],
]

IsingInstance = exp5573.IsingInstance

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5611_cdls_matched_sampler_crossover.json")
DESCRIPTOR_SOURCE_RELATIVE_PATH = exp5573.DESCRIPTOR_SOURCE_RELATIVE_PATH

EXPERIMENT = 5611
EXPERIMENT_ID = "exp5611-cdls-matched-sampler-crossover"
MILESTONE = "2026.07.506"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5611.cdls_matched_sampler_crossover.v1"
SPEC_REFS = ("REQ-SAMPLE-5611", "SCENARIO-SAMPLE-5611")
INFERENCE_SUBSTRATE = "matched_cpu_cuda_exact_ising_sampling"

DEFAULT_INSTANCE_SIZES = (128, 256, 512, 1024)
DEFAULT_SEEDS = (5611,)
DEFAULT_SAMPLES_PER_PAIR = 10_000
DEFAULT_WARMUP_STEPS = exp5573.DEFAULT_WARMUP_STEPS
DEFAULT_THINNING = exp5573.DEFAULT_THINNING
DEFAULT_TEMPERATURE = exp5573.DEFAULT_TEMPERATURE
DEFAULT_PRECISION = exp5573.DEFAULT_PRECISION
DEFAULT_STOPPING_RULE = exp5573.DEFAULT_STOPPING_RULE
DEFAULT_CDLS_CONTINUOUS_BOUND = 3.0
DEFAULT_CDLS_PROPOSAL_STD = 0.35
DEFAULT_CDLS_DRIFT_SCALE = 0.08
MIN_SAMPLES_PER_SUCCESS = 10_000
MIN_SEEDS_FOR_TIMING_INTERVAL = 3
TERMINAL_PREFIXES = ("complete:", "blocked:")

METHODS: tuple[JsonDict, ...] = (
    {
        "method_id": "discrete_dls",
        "label": "Exp5573 discrete DLS heat-bath baseline",
        "baseline_preserved": True,
        "projection_correction": "not_applicable_discrete_baseline",
    },
    {
        "method_id": "bounded_cdls",
        "label": "Bounded cDLS continuous-intermediate proposal",
        "baseline_preserved": False,
        "projection_correction": "metropolis_hastings_exact_discrete_target",
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every headline and gate field exists before a reviewer trusts the JSON shape.",
    "target_descriptors": "Proves both methods sample identical descriptor-derived Ising problems.",
    "instance_sizes": "Makes the tested crossover range explicit instead of implying unmeasured scale.",
    "methods": "Keeps the unchanged discrete DLS baseline and bounded cDLS proposal separately identifiable.",
    "seeds": "Records paired seeds so every CPU/CUDA and method comparison is reproducible.",
    "samples_per_pair": "Guards the >=10000 retained-sample floor required for mixing estimates.",
    "cpu_device_receipt": "Authenticates CPU identity, runtime, and free memory for the local benchmark.",
    "cuda_device_receipt": "Authenticates CUDA identity, driver/runtime, and free memory or records a precise blocker.",
    "timing_rows": "Preserves raw matched timing evidence, including failed rows, before any summary ratio.",
    "energy_quality_metrics": "Prevents speed from hiding wrong samples by reporting energy and exact constraint quality.",
    "mixing_metrics": "Reports ESS and autocorrelation to test whether the cDLS mechanism actually mixes.",
    "successful_matched_pairs": "Counts only complete method/device rows that satisfy the same schedule; quality gates are recorded before claims.",
    "speedup_by_pair": "Reports only matched ratios; failed or quality-inferior rows cannot enter speedups.",
    "crossover_size": "Records the smallest gated crossover size, or null when no crossover is proven.",
    "crossover_claim_allowed": "Requires quality and timing gates to pass together before any crossover claim.",
    "board_speedup_claimed": "Bare false prevents this CPU/CUDA sampler study from reopening board continuity.",
    "inference_substrate": "Declares matched CPU/CUDA exact Ising sampling, not LLM inference or board timing.",
    "honest_verdict": "Terminal complete: or blocked: verdict states whether no-crossover evidence is final.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for content-addressed benchmark evidence."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash text with SHA-256 using the repository's standard hex digest."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content after canonical serialization."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_ising_instances(
    descriptor_payload: Mapping[str, Any],
    instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES,
) -> list[IsingInstance]:
    """Build descriptor-derived Ising instances through the existing Exp5573 path."""

    return exp5573.build_ising_instances(descriptor_payload, instance_sizes)


def matched_schedule_for_instances(instances: Sequence[IsingInstance]) -> JsonDict:
    """Return the shared DLS/cDLS schedule and exact target checksums."""

    return {
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "stopping_rule": DEFAULT_STOPPING_RULE,
        "couplings_biases_shared": True,
        "measurement_boundaries_shared": True,
        "cdls_continuous_bound": DEFAULT_CDLS_CONTINUOUS_BOUND,
        "cdls_proposal_std": DEFAULT_CDLS_PROPOSAL_STD,
        "cdls_drift_scale": DEFAULT_CDLS_DRIFT_SCALE,
        "instance_checksums": {instance.instance_id: instance.checksum for instance in instances},
    }


def target_descriptors_for_instances(instances: Sequence[IsingInstance]) -> list[JsonDict]:
    """Describe the exact Ising targets shared by both sampler methods."""

    descriptors: list[JsonDict] = []
    for instance in instances:
        descriptors.append(
            {
                "instance_id": instance.instance_id,
                "size": int(instance.size),
                "descriptor_ids": list(instance.descriptor_ids),
                "descriptor_checksum": instance.checksum,
                "couplings_checksum": sha256_json(np.round(instance.couplings, 6).tolist()),
                "biases_checksum": sha256_json(np.round(instance.biases, 6).tolist()),
                "target_spins_checksum": sha256_json(instance.target_spins.astype(int).tolist()),
                "constraint_count": len(instance.constraint_indices),
            }
        )
    return descriptors


def quality_equivalence_gate() -> JsonDict:
    """Define non-inferiority thresholds before timing summaries are built."""

    return {
        "defined_before_timing": True,
        "energy_mean_tolerance_abs": 0.5,
        "energy_mean_tolerance_rel": 0.05,
        "best_energy_tolerance_abs": 0.5,
        "constraint_satisfaction_tolerance_abs": 0.03,
        "min_effective_sample_size": 100.0,
        "min_seeds_for_timing_interval": MIN_SEEDS_FOR_TIMING_INTERVAL,
        "timing_interval": "observed_seed_min_max",
    }


def run_matched_sampler_rows(
    instances: list[IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: JsonDict,
    torch_module: Any,
    clock: Clock = time.perf_counter,
) -> list[JsonDict]:  # pragma: no cover - exercised by the live experiment command.
    """Run DLS and cDLS on CPU/CUDA while preserving failed rows explicitly."""

    rows: list[JsonDict] = []
    for instance in instances:
        for seed in seeds:
            for backend in ("cpu", "cuda"):
                for method in ("discrete_dls", "bounded_cdls"):
                    try:
                        if method == "discrete_dls":
                            row = run_discrete_baseline_row(
                                instance=instance,
                                backend=backend,
                                seed=seed,
                                samples_per_pair=samples_per_pair,
                                matched_schedule=matched_schedule,
                                torch_module=torch_module,
                                clock=clock,
                            )
                        else:
                            row = run_cdls_sampler_row(
                                instance=instance,
                                backend=backend,
                                seed=seed,
                                samples_per_pair=samples_per_pair,
                                matched_schedule=matched_schedule,
                                torch_module=torch_module,
                                clock=clock,
                            )
                    except RuntimeError as exc:
                        if _is_oom_error(exc):
                            row = blocked_timing_row(instance, seed, backend, method, "oom")
                        else:
                            row = blocked_timing_row(
                                instance,
                                seed,
                                backend,
                                method,
                                f"runtime_error:{type(exc).__name__}",
                            )
                    except Exception as exc:  # noqa: BLE001 - benchmark rows preserve blockers.
                        row = blocked_timing_row(
                            instance,
                            seed,
                            backend,
                            method,
                            f"failed:{type(exc).__name__}",
                        )
                    rows.append(row)
    return rows


def run_discrete_baseline_row(
    *,
    instance: IsingInstance,
    backend: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    torch_module: Any,
    clock: Clock = time.perf_counter,
) -> JsonDict:  # pragma: no cover - the unchanged baseline is covered by Exp5573 tests.
    """Call the Exp5573 baseline unchanged and annotate the method boundary."""

    memory_before = memory_snapshot(torch_module, backend)
    row = exp5573.run_one_sampler_row(
        instance=instance,
        backend=backend,
        seed=seed,
        samples_per_pair=samples_per_pair,
        matched_schedule=matched_schedule,
        torch_module=torch_module,
        clock=clock,
    )
    row.update(
        {
            "method": "discrete_dls",
            "device": backend,
            "acceptance_rate": 1.0,
            "exact_constraint_satisfaction_rate": row["constraint_satisfaction_rate"],
            "compile_time_s": 0.0,
            "end_to_end_wall_time_s": row["wall_time_s"],
            "memory_before": memory_before,
            "memory_after": memory_snapshot(torch_module, backend),
            "kernel_device_path": f"torch_{backend}_exp5573_heat_bath",
            "projection_correction": "not_applicable_discrete_baseline",
        }
    )
    return row


def run_cdls_sampler_row(
    *,
    instance: IsingInstance,
    backend: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    torch_module: Any,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Run one bounded cDLS row with exact projection correction.

    The proposal distribution is Gaussian in a continuous intermediate. Because
    only the projected sign state is retained, the Metropolis-Hastings ratio is
    computed from the probability that the Gaussian intermediate lands on the
    projected side for each spin. That correction is what keeps the accepted
    chain targeted at the exact discrete Ising distribution instead of the
    relaxed continuous proposal.
    """

    torch = torch_module
    device = torch.device("cuda:0" if backend == "cuda" else "cpu")
    dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    couplings = torch.tensor(instance.couplings, device=device, dtype=dtype)
    biases = torch.tensor(instance.biases, device=device, dtype=dtype)
    target = torch.tensor(instance.target_spins, device=device, dtype=dtype)
    constraint_indices = torch.tensor(instance.constraint_indices, device=device, dtype=torch.long)
    beta = float(1.0 / float(matched_schedule["temperature"]))
    continuous_bound = float(matched_schedule["cdls_continuous_bound"])
    proposal_std = float(matched_schedule["cdls_proposal_std"])
    drift_scale = float(matched_schedule["cdls_drift_scale"])

    spins = torch.where(
        torch.rand(instance.size, device=device, generator=generator) < 0.5,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    _sync_if_cuda(torch, backend)
    memory_before = memory_snapshot(torch, backend)
    compile_time_s = 0.0
    accepted = 0
    rejected = 0
    max_abs_continuous = 0.0

    warmup_start = clock()
    for _ in range(int(matched_schedule["warmup_steps"])):
        spins, step_accepted, step_abs = _cdls_step(
            torch,
            spins,
            couplings,
            biases,
            beta,
            generator,
            continuous_bound,
            proposal_std,
            drift_scale,
        )
        accepted += int(step_accepted)
        rejected += int(not step_accepted)
        max_abs_continuous = max(max_abs_continuous, step_abs)
    _sync_if_cuda(torch, backend)
    warmup_time_s = max(clock() - warmup_start, 0.0)

    energies: list[float] = []
    satisfaction: list[float] = []
    sample_start = clock()
    thinning = int(matched_schedule["thinning"])
    total_steps = int(samples_per_pair) * thinning
    for step in range(total_steps):
        spins, step_accepted, step_abs = _cdls_step(
            torch,
            spins,
            couplings,
            biases,
            beta,
            generator,
            continuous_bound,
            proposal_std,
            drift_scale,
        )
        accepted += int(step_accepted)
        rejected += int(not step_accepted)
        max_abs_continuous = max(max_abs_continuous, step_abs)
        if (step + 1) % thinning == 0:
            energy = exp5573._ising_energy(torch, spins, couplings, biases)
            energies.append(float(energy.detach().cpu().item()))
            satisfied = (spins[constraint_indices] == target[constraint_indices]).to(dtype).mean()
            satisfaction.append(float(satisfied.detach().cpu().item()))
    _sync_if_cuda(torch, backend)
    sample_time_s = max(clock() - sample_start, 0.0)

    energy_arr = np.asarray(energies, dtype=np.float64)
    satisfaction_arr = np.asarray(satisfaction, dtype=np.float64)
    tau = exp5573.integrated_autocorrelation_time(energy_arr)
    ess = float(samples_per_pair / tau) if tau > 0 else float(samples_per_pair)
    proposals = accepted + rejected
    wall_time_s = float(compile_time_s + warmup_time_s + sample_time_s)
    return {
        "status": "success",
        "pair_id": f"{instance.instance_id}:seed{seed}",
        "method": "bounded_cdls",
        "backend": backend,
        "device": backend,
        "instance_id": instance.instance_id,
        "size": instance.size,
        "seed": int(seed),
        "samples": int(samples_per_pair),
        "temperature": float(matched_schedule["temperature"]),
        "warmup_steps": int(matched_schedule["warmup_steps"]),
        "thinning": thinning,
        "precision": str(matched_schedule["precision"]),
        "best_energy": round(float(np.min(energy_arr)), 8),
        "energy_mean": round(float(np.mean(energy_arr)), 8),
        "energy_std": round(float(np.std(energy_arr)), 8),
        "energy_min": round(float(np.min(energy_arr)), 8),
        "energy_max": round(float(np.max(energy_arr)), 8),
        "energy_quantiles": {
            "p05": round(float(np.quantile(energy_arr, 0.05)), 8),
            "p50": round(float(np.quantile(energy_arr, 0.50)), 8),
            "p95": round(float(np.quantile(energy_arr, 0.95)), 8),
        },
        "constraint_satisfaction_rate": round(float(np.mean(satisfaction_arr)), 8),
        "exact_constraint_satisfaction_rate": round(float(np.mean(satisfaction_arr)), 8),
        "acceptance_rate": round(float(accepted / proposals), 8) if proposals else 0.0,
        "mh_accept_count": accepted,
        "mh_reject_count": rejected,
        "proposal_count": proposals,
        "continuous_bound": continuous_bound,
        "continuous_bound_observed_abs_max": round(float(max_abs_continuous), 8),
        "cdls_proposal_std": proposal_std,
        "cdls_drift_scale": drift_scale,
        "autocorrelation_time": round(float(tau), 8),
        "effective_sample_size": round(float(ess), 8),
        "compile_time_s": round(float(compile_time_s), 8),
        "wall_time_s": round(wall_time_s, 8),
        "end_to_end_wall_time_s": round(wall_time_s, 8),
        "warmup_time_s": round(float(warmup_time_s), 8),
        "sample_time_s": round(float(sample_time_s), 8),
        "memory_before": memory_before,
        "memory_after": memory_snapshot(torch, backend),
        "kernel_device_path": f"torch_{backend}_bounded_cdls_dense_matvec_mh",
        "projection_correction": "metropolis_hastings_exact_discrete_target",
        "result_hash": sha256_json(
            {
                "instance_id": instance.instance_id,
                "backend": backend,
                "method": "bounded_cdls",
                "seed": seed,
                "energies": [round(float(value), 6) for value in energy_arr.tolist()],
                "satisfaction": [round(float(value), 6) for value in satisfaction_arr.tolist()],
                "accepted": accepted,
                "rejected": rejected,
            }
        ),
    }


def _cdls_step(
    torch: Any,
    spins: Any,
    couplings: Any,
    biases: Any,
    beta: float,
    generator: Any,
    continuous_bound: float,
    proposal_std: float,
    drift_scale: float,
) -> tuple[Any, bool, float]:
    field = torch.matmul(couplings, spins) + biases
    mean = spins + float(drift_scale) * float(beta) * field
    noise = torch.randn(spins.shape, device=spins.device, generator=generator, dtype=spins.dtype)
    continuous = torch.clamp(mean + float(proposal_std) * noise, -continuous_bound, continuous_bound)
    proposed = torch.where(continuous >= 0.0, torch.ones_like(spins), -torch.ones_like(spins))

    current_energy = exp5573._ising_energy(torch, spins, couplings, biases)
    proposed_energy = exp5573._ising_energy(torch, proposed, couplings, biases)
    log_forward = _log_projected_proposal_probability(
        torch,
        projected=proposed,
        source=spins,
        couplings=couplings,
        biases=biases,
        beta=beta,
        proposal_std=proposal_std,
        drift_scale=drift_scale,
    )
    log_reverse = _log_projected_proposal_probability(
        torch,
        projected=spins,
        source=proposed,
        couplings=couplings,
        biases=biases,
        beta=beta,
        proposal_std=proposal_std,
        drift_scale=drift_scale,
    )
    log_accept = -float(beta) * (proposed_energy - current_energy) + log_reverse - log_forward
    uniform = torch.rand((), device=spins.device, generator=generator, dtype=spins.dtype)
    accepted = bool((torch.log(uniform.clamp_min(1e-12)) < torch.minimum(log_accept, torch.zeros_like(log_accept))).item())
    next_spins = proposed if accepted else spins
    return next_spins, accepted, float(torch.max(torch.abs(continuous)).detach().cpu().item())


def _log_projected_proposal_probability(
    torch: Any,
    *,
    projected: Any,
    source: Any,
    couplings: Any,
    biases: Any,
    beta: float,
    proposal_std: float,
    drift_scale: float,
) -> Any:
    field = torch.matmul(couplings, source) + biases
    mean = source + float(drift_scale) * float(beta) * field
    z = projected * mean / (float(proposal_std) * sqrt(2.0))
    probabilities = 0.5 * torch.erfc(-z)
    return torch.log(probabilities.clamp_min(1e-12)).sum()


def _sync_if_cuda(torch: Any, backend: str) -> None:
    if backend == "cuda":
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.synchronize()


def memory_snapshot(torch_module: Any, backend: str) -> JsonDict:
    """Return a compact free-memory snapshot for CPU or CUDA timing rows."""

    if backend == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return {"status": "blocked", "blocked_reason": "cuda_unavailable"}
        try:
            free_bytes, total_bytes = cuda.mem_get_info(0)
            return {
                "status": "reachable",
                "free_mib": int(free_bytes) // (1024 * 1024),
                "total_mib": int(total_bytes) // (1024 * 1024),
            }
        except Exception as exc:  # noqa: BLE001 - memory APIs vary across fakes.
            return {"status": "blocked", "blocked_reason": type(exc).__name__}
    memory = exp5573.parse_meminfo(exp5573._read_meminfo())
    return {"status": "reachable", **memory}


def blocked_timing_row(
    instance: IsingInstance,
    seed: int,
    backend: str,
    method: str,
    reason: str,
) -> JsonDict:
    """Create an explicit failed row that cannot enter speedup summaries."""

    return {
        "status": "blocked",
        "blocked_reason": reason,
        "pair_id": f"{instance.instance_id}:seed{seed}",
        "method": method,
        "backend": backend,
        "device": backend,
        "instance_id": instance.instance_id,
        "size": int(instance.size),
        "seed": int(seed),
        "samples": 0,
        "wall_time_s": None,
        "warmup_time_s": None,
        "sample_time_s": None,
        "compile_time_s": None,
        "end_to_end_wall_time_s": None,
        "kernel_device_path": f"torch_{backend}_{method}",
    }


def _is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower() or "cuda error: out of memory" in str(exc).lower()


def summarize_rows(raw_rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Summarize matched quality, mixing, timing, and speedup rows."""

    grouped: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for row in raw_rows:
        pair_id = str(row.get("pair_id", ""))
        method = str(row.get("method", ""))
        backend = str(row.get("backend", ""))
        grouped.setdefault(pair_id, {})[(method, backend)] = row

    energy_metrics: list[JsonDict] = []
    mixing_metrics: list[JsonDict] = []
    timing_rows: list[JsonDict] = []
    speedups: list[JsonDict] = []
    for pair_id, rows in sorted(grouped.items()):
        for row in rows.values():
            timing_rows.append(_timing_summary(row))
        required_keys = {
            ("discrete_dls", "cpu"),
            ("discrete_dls", "cuda"),
            ("bounded_cdls", "cpu"),
            ("bounded_cdls", "cuda"),
        }
        if not required_keys.issubset(rows):
            continue
        if any(rows[key].get("status", "success") != "success" for key in required_keys):
            continue
        if not _all_rows_match([rows[key] for key in required_keys]):
            continue

        dls_cpu = rows[("discrete_dls", "cpu")]
        dls_cuda = rows[("discrete_dls", "cuda")]
        cdls_cpu = rows[("bounded_cdls", "cpu")]
        cdls_cuda = rows[("bounded_cdls", "cuda")]
        quality_by_device = {
            "cpu": _quality_noninferior(dls_cpu, cdls_cpu),
            "cuda": _quality_noninferior(dls_cuda, cdls_cuda),
        }
        quality_noninferior = all(quality_by_device.values())
        energy_metrics.append(
            {
                "pair_id": pair_id,
                "instance_id": dls_cpu["instance_id"],
                "size": int(dls_cpu["size"]),
                "seed": int(dls_cpu["seed"]),
                "quality_noninferior": quality_noninferior,
                "quality_by_device": quality_by_device,
                "cpu": {
                    "discrete_dls": _energy_summary(dls_cpu),
                    "bounded_cdls": _energy_summary(cdls_cpu),
                },
                "cuda": {
                    "discrete_dls": _energy_summary(dls_cuda),
                    "bounded_cdls": _energy_summary(cdls_cuda),
                },
            }
        )
        mixing_metrics.append(
            {
                "pair_id": pair_id,
                "instance_id": dls_cpu["instance_id"],
                "size": int(dls_cpu["size"]),
                "seed": int(dls_cpu["seed"]),
                "cpu": {
                    "discrete_dls": _mixing_summary(dls_cpu),
                    "bounded_cdls": _mixing_summary(cdls_cpu),
                },
                "cuda": {
                    "discrete_dls": _mixing_summary(dls_cuda),
                    "bounded_cdls": _mixing_summary(cdls_cuda),
                },
            }
        )
        speedups.append(
            {
                "pair_id": pair_id,
                "instance_id": dls_cpu["instance_id"],
                "size": int(dls_cpu["size"]),
                "seed": int(dls_cpu["seed"]),
                "quality_noninferior": quality_noninferior,
                "quality_by_device": quality_by_device,
                "discrete_dls_cuda_vs_cpu_speedup": _ratio(dls_cpu, dls_cuda),
                "bounded_cdls_cuda_vs_cpu_speedup": _ratio(cdls_cpu, cdls_cuda),
                "bounded_cdls_vs_discrete_dls_cpu_speedup": _ratio(dls_cpu, cdls_cpu),
                "bounded_cdls_vs_discrete_dls_cuda_speedup": _ratio(dls_cuda, cdls_cuda),
                "wall_time_s": {
                    "cpu_discrete_dls": float(dls_cpu["wall_time_s"]),
                    "cpu_bounded_cdls": float(cdls_cpu["wall_time_s"]),
                    "cuda_discrete_dls": float(dls_cuda["wall_time_s"]),
                    "cuda_bounded_cdls": float(cdls_cuda["wall_time_s"]),
                },
            }
        )
    return energy_metrics, mixing_metrics, timing_rows, speedups


def _all_rows_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    keys = ("instance_id", "size", "seed", "samples", "temperature", "warmup_steps", "thinning", "precision")
    first = rows[0]
    return all(row.get(key) == first.get(key) for row in rows for key in keys)


def _quality_noninferior(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    gate = quality_equivalence_gate()
    energy_tol = max(
        float(gate["energy_mean_tolerance_abs"]),
        float(gate["energy_mean_tolerance_rel"]) * abs(float(baseline["energy_mean"])),
    )
    best_tol = float(gate["best_energy_tolerance_abs"])
    constraint_tol = float(gate["constraint_satisfaction_tolerance_abs"])
    return bool(
        float(candidate["energy_mean"]) <= float(baseline["energy_mean"]) + energy_tol
        and float(candidate["best_energy"]) <= float(baseline["best_energy"]) + best_tol
        and float(candidate["exact_constraint_satisfaction_rate"])
        >= float(baseline["exact_constraint_satisfaction_rate"]) - constraint_tol
        and float(candidate["effective_sample_size"]) >= float(gate["min_effective_sample_size"])
    )


def _ratio(numerator_row: Mapping[str, Any], denominator_row: Mapping[str, Any]) -> float | None:
    denominator = float(denominator_row["wall_time_s"])
    if denominator <= 0.0:
        return None
    return round(float(numerator_row["wall_time_s"]) / denominator, 8)


def _energy_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "best_energy": float(row["best_energy"]),
        "mean_energy": float(row["energy_mean"]),
        "energy_std": float(row["energy_std"]),
        "energy_min": float(row["energy_min"]),
        "energy_max": float(row["energy_max"]),
        "energy_quantiles": dict(row["energy_quantiles"]),
        "acceptance_rate": float(row["acceptance_rate"]),
        "exact_constraint_satisfaction_rate": float(row["exact_constraint_satisfaction_rate"]),
    }


def _mixing_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "integrated_autocorrelation_time": float(row["autocorrelation_time"]),
        "effective_sample_size": float(row["effective_sample_size"]),
    }


def _timing_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "status": row.get("status", "success"),
        "blocked_reason": row.get("blocked_reason"),
        "pair_id": row["pair_id"],
        "method": row["method"],
        "backend": row["backend"],
        "device": row.get("device", row["backend"]),
        "instance_id": row["instance_id"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "samples": int(row.get("samples", 0)),
        "wall_time_s": row.get("wall_time_s"),
        "end_to_end_wall_time_s": row.get("end_to_end_wall_time_s"),
        "compile_time_s": row.get("compile_time_s"),
        "warmup_time_s": row.get("warmup_time_s"),
        "sample_time_s": row.get("sample_time_s"),
        "memory_before": row.get("memory_before"),
        "memory_after": row.get("memory_after"),
        "kernel_device_path": row.get("kernel_device_path"),
        "result_hash": row.get("result_hash"),
    }


def crossover_from_speedups(speedups: Sequence[Mapping[str, Any]]) -> tuple[int | None, bool, list[JsonDict]]:
    """Return the smallest cDLS CUDA-vs-CPU crossover size allowed by gates."""

    intervals: list[JsonDict] = []
    by_size: dict[int, list[float]] = {}
    for row in speedups:
        if row.get("quality_noninferior") is not True:
            continue
        value = row.get("bounded_cdls_cuda_vs_cpu_speedup")
        if value is None:
            continue
        by_size.setdefault(int(row["size"]), []).append(float(value))
    for size in sorted(by_size):
        values = by_size[size]
        interval = {
            "size": size,
            "n_seed_pairs": len(values),
            "speedup_min": round(min(values), 8),
            "speedup_max": round(max(values), 8),
            "excludes_1_favorable": len(values) >= MIN_SEEDS_FOR_TIMING_INTERVAL and min(values) > 1.0,
        }
        intervals.append(interval)
        if interval["excludes_1_favorable"]:
            return size, True, intervals
    return None, False, intervals


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    torch_module: Any | None = None,
    sampler_runner: SamplerRunner = run_matched_sampler_rows,
    clock: Clock = time.perf_counter,
    instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5611 terminal artifact."""

    started = clock()
    descriptor_source = exp5573.load_descriptor_source(root)
    torch_obj = torch_module
    if torch_obj is None:
        try:
            torch_obj = exp5573._import_torch()
        except Exception:
            torch_obj = None
    cpu_receipt = exp5573.cpu_device_receipt(torch_obj)
    cuda_receipt = exp5573.cuda_device_receipt(torch_obj)
    preconditions: JsonDict = {
        "cpu_available": cpu_receipt["status"] == "reachable",
        "cuda_available": cuda_receipt["status"] == "reachable",
        "descriptor_available": descriptor_source["available"],
        "blocked_reasons": [],
    }
    if not descriptor_source["available"]:
        preconditions["blocked_reasons"].append(descriptor_source["blocked_reason"])
    if cuda_receipt["status"] != "reachable":
        preconditions["blocked_reasons"].append(cuda_receipt.get("blocked_reason", "cuda_blocked"))

    instances: list[IsingInstance] = []
    raw_rows: list[JsonDict] = []
    matched_schedule: JsonDict = {
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "stopping_rule": DEFAULT_STOPPING_RULE,
        "couplings_biases_shared": False,
        "measurement_boundaries_shared": False,
        "instance_checksums": {},
    }
    if descriptor_source["available"]:
        try:
            instances = build_ising_instances(descriptor_source["payload"], instance_sizes)
            matched_schedule = matched_schedule_for_instances(instances)
        except Exception as exc:  # noqa: BLE001 - artifact records descriptor build blockers.
            preconditions["descriptor_available"] = False
            preconditions["blocked_reasons"].append(f"descriptor_instance_build_failed:{type(exc).__name__}")

    if (
        preconditions["cpu_available"]
        and preconditions["cuda_available"]
        and preconditions["descriptor_available"]
        and torch_obj is not None
    ):
        try:
            raw_rows = sampler_runner(
                instances,
                tuple(int(seed) for seed in seeds),
                int(samples_per_pair),
                matched_schedule,
                torch_obj,
                clock,
            )
        except Exception as exc:  # noqa: BLE001 - block instead of synthesizing rows.
            preconditions["blocked_reasons"].append(f"sampler_run_failed:{type(exc).__name__}:{exc}")
            raw_rows = []

    energy_metrics, mixing_metrics, timing_rows, speedups = summarize_rows(raw_rows)
    crossover_size, crossover_allowed, timing_intervals = crossover_from_speedups(speedups)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions": preconditions,
        "descriptor_source": {
            "path": descriptor_source["path"],
            "available": descriptor_source["available"],
            "sha256": descriptor_source["sha256"],
            "descriptor_count": descriptor_source["descriptor_count"],
            "blocked_reason": descriptor_source["blocked_reason"],
        },
        "target_descriptors": target_descriptors_for_instances(instances),
        "instance_sizes": [int(size) for size in instance_sizes],
        "methods": [dict(method) for method in METHODS],
        "seeds": [int(seed) for seed in seeds],
        "samples_per_pair": int(samples_per_pair),
        "matched_schedule": matched_schedule,
        "quality_equivalence_gate": quality_equivalence_gate(),
        "cpu_device_receipt": cpu_receipt,
        "cuda_device_receipt": cuda_receipt,
        "timing_rows": timing_rows,
        "energy_quality_metrics": energy_metrics,
        "mixing_metrics": mixing_metrics,
        "successful_matched_pairs": len(speedups),
        "speedup_by_pair": speedups,
        "timing_intervals_by_size": timing_intervals,
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


def _normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    if tests_added_or_reused:
        return [str(item) for item in tests_added_or_reused]
    return [
        "tests/python/test_experiment_5611_cdls_matched_sampler_crossover.py",
        ".venv/bin/pytest tests/python -q",
    ]


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict without implying a board or crossover win."""

    blockers = list(payload.get("preconditions", {}).get("blocked_reasons", []))
    successful = int(payload.get("successful_matched_pairs", 0))
    if successful > 0:
        if payload.get("crossover_claim_allowed") is True:
            return (
                "complete: cDLS matched CPU/CUDA evidence recorded; "
                f"crossover_size={payload.get('crossover_size')}; board_speedup_claimed=false"
            )
        return (
            "complete: cDLS matched CPU/CUDA evidence recorded with no gated crossover; "
            f"successful_matched_pairs={successful}; board_speedup_claimed=false"
        )
    return (
        "blocked: cDLS matched CPU/CUDA comparison unavailable; "
        f"blocked_reasons={len(blockers)}; board_speedup_claimed=false"
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate required fields and no-overclaim boundaries for Exp5611."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")  # pragma: no cover
    if int(payload.get("samples_per_pair", 0)) < MIN_SAMPLES_PER_SUCCESS:
        raise ValueError("samples_per_pair must be at least 10000")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    if payload.get("board_speedup_claimed") is not False:
        raise ValueError("board_speedup_claimed must be false")
    speedups = payload.get("speedup_by_pair")
    if not isinstance(speedups, list):
        raise ValueError("speedup_by_pair must be a list")  # pragma: no cover
    if int(payload.get("successful_matched_pairs", -1)) != len(speedups):
        raise ValueError("successful_matched_pairs must match speedup_by_pair length")  # pragma: no cover
    if payload.get("crossover_claim_allowed") is True:
        if payload.get("crossover_size") is None:
            raise ValueError("crossover_claim_allowed requires crossover_size")
        if not speedups or any(row.get("quality_noninferior") is not True for row in speedups):
            raise ValueError("crossover_claim_allowed requires quality_noninferior speedups")
        intervals = payload.get("timing_intervals_by_size")
        if not isinstance(intervals, list) or not any(
            row.get("size") == payload.get("crossover_size") and row.get("excludes_1_favorable") is True
            for row in intervals
            if isinstance(row, Mapping)
        ):
            raise ValueError("crossover_claim_allowed requires favorable timing interval")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")  # pragma: no cover
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact with stable formatting."""

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
    """Build, validate, and write Exp5611 outputs."""

    artifact = build_artifact(root=repo_root, tests_added_or_reused=tests_added_or_reused)
    return write_output(repo_root, artifact)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
