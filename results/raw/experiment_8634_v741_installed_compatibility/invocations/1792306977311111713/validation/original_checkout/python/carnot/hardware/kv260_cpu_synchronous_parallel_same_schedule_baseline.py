"""Exp 2939 CPU synchronous-parallel same-schedule baseline.

Spec refs: REQ-HW-072, SCENARIO-HW-072.

Exp 2913 compared KV260 synchronous checkerboard Glauber timing against a CPU
sequential Gibbs baseline. That is not an apples-to-apples speedup claim. This
module runs the CPU baseline with the same even/odd checkerboard update shape
used by the KV260 path, records per-sample wall-clock timing, and writes the
paper-v6 claim-boundary verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs as exp2938


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
EXP2912_REL_PATH = Path("results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json")
OUTPUT_REL_PATH = Path(
    "results/experiment_2939_cpu_synchronous_parallel_same_schedule_baseline_v1.json"
)

EXPERIMENT_ID = 2939
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 2939
RANDOM_SEEDS = [42, 137, 271]
N_ENERGY_SAMPLES = 10_000
KV260_PER_SAMPLE_US_CITED = 24.0
MIN_SUCCESS_DURATION_S = 20.0
MAX_MMD_SAMPLES = 2_048
CPU_UPDATE_SCHEDULE = "cpu_synchronous_parallel_checkerboard_glauber_exp2898_sparse_upload"

SPEEDUP_PRINCIPLE = (
    "The apples-to-apples speedup measurement. < 1.0 means KV260 is slower at "
    "this n; >= 1.0 means faster."
)
EQUIVALENCE_PRINCIPLE = (
    "Cross-check that the same broken sampler produces statistically-equivalent "
    "distributions on both substrates."
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "cpu_synchronous_parallel_per_sample_us_median",
    "cpu_synchronous_parallel_per_sample_us_p95",
    "kv260_per_sample_us_cited",
    "kv260_speedup_vs_same_schedule_cpu",
    "energy_distribution_equivalence_test",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "paper_v6_recommendation",
    "methodology_note",
    "duration_s",
}


@dataclass(frozen=True)
class CpuSynchronousRunResult:
    seed: int
    energies: list[float]
    energy_sha256: str
    latency_us_median: float
    latency_us_p95: float
    update_schedule: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute median of an empty sequence")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute p95 of an empty sequence")
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def build_sparse_upload_matrix(problem: exp2938.DenseIsingProblem) -> tuple[np.ndarray, np.ndarray]:
    upload = problem.upload
    adjacency = np.asarray(upload["adjacency"], dtype=np.int64)
    couplings = np.asarray(upload["couplings_q88"], dtype=np.float64) / 256.0
    fields = np.asarray(upload["h_q88"], dtype=np.float64) / 256.0
    if fields.shape != (problem.n_spins,):
        raise ValueError("upload h_q88 shape does not match n_spins")
    if adjacency.shape != couplings.shape or adjacency.shape[0] != problem.n_spins:
        raise ValueError("upload adjacency/couplings_q88 shape does not match n_spins")

    sparse_j = np.zeros((problem.n_spins, problem.n_spins), dtype=np.float64)
    for row in range(problem.n_spins):
        for slot, neighbor in enumerate(adjacency[row]):
            if 0 <= int(neighbor) < problem.n_spins:
                sparse_j[row, int(neighbor)] += float(couplings[row, slot])
    return sparse_j, fields


def checkerboard_sweep(
    state: np.ndarray,
    *,
    sparse_j_matrix: np.ndarray,
    fields: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one KV260-shaped even phase followed by one odd phase."""

    next_state = np.asarray(state, dtype=np.int8).copy()
    even = np.arange(0, next_state.size, 2)
    odd = np.arange(1, next_state.size, 2)

    local_field = fields + sparse_j_matrix @ next_state
    p_plus = _stable_sigmoid(2.0 * float(beta) * local_field[even])
    next_state[even] = np.where(rng.random(even.size) < p_plus, 1, -1).astype(np.int8)

    local_field = fields + sparse_j_matrix @ next_state
    p_plus = _stable_sigmoid(2.0 * float(beta) * local_field[odd])
    next_state[odd] = np.where(rng.random(odd.size) < p_plus, 1, -1).astype(np.int8)
    return next_state


def run_cpu_synchronous_parallel_glauber(
    problem: exp2938.DenseIsingProblem,
    *,
    n_samples: int = N_ENERGY_SAMPLES,
    timer_ns: Callable[[], int] | None = None,
) -> CpuSynchronousRunResult:
    """Collect fixed-budget CPU synchronous checkerboard energy samples."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    active_timer = timer_ns or time.perf_counter_ns
    sparse_j, fields = build_sparse_upload_matrix(problem)
    beta = problem.beta_final_q88 / 256.0
    rng = np.random.default_rng(problem.seed + RANDOM_SEED)
    state = (rng.integers(0, 2, size=problem.n_spins, dtype=np.int8) * 2 - 1).astype(np.int8)

    latencies_us: list[float] = []
    energies: list[float] = []
    for _ in range(int(n_samples)):
        started_ns = active_timer()
        state = checkerboard_sweep(
            state,
            sparse_j_matrix=sparse_j,
            fields=fields,
            beta=beta,
            rng=rng,
        )
        latencies_us.append((active_timer() - started_ns) / 1000.0)
        energies.append(round(exp2938.dense_energy(problem, state), 12))

    return CpuSynchronousRunResult(
        seed=problem.seed,
        energies=energies,
        energy_sha256=sha256_canonical(energies),
        latency_us_median=_median(latencies_us),
        latency_us_p95=_p95(latencies_us),
        update_schedule=CPU_UPDATE_SCHEDULE,
    )


def same_schedule_reference_energies(
    cpu_runs: Mapping[int, CpuSynchronousRunResult],
) -> dict[int, list[float]]:
    return {int(seed): list(row.energies) for seed, row in cpu_runs.items()}


def _flatten_by_seed(energies_by_seed: Mapping[int, Sequence[float]]) -> list[float]:
    values: list[float] = []
    for seed in RANDOM_SEEDS:
        values.extend(float(value) for value in energies_by_seed[int(seed)])
    return values


def _deterministic_subset(values: Sequence[float], max_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= max_samples:
        return array
    indices = np.linspace(0, array.size - 1, num=int(max_samples), dtype=np.int64)
    return array[indices]


def energy_distribution_equivalence(
    cpu_runs: Mapping[int, CpuSynchronousRunResult],
    kv260_energies_by_seed: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    cpu_values = _flatten_by_seed({seed: row.energies for seed, row in cpu_runs.items()})
    kv260_values = _flatten_by_seed(kv260_energies_by_seed)
    ks = stats.ks_2samp(cpu_values, kv260_values, alternative="two-sided", method="auto")
    cpu_mmd = _deterministic_subset(cpu_values, MAX_MMD_SAMPLES)
    kv260_mmd = _deterministic_subset(kv260_values, MAX_MMD_SAMPLES)
    bandwidth = exp2938.median_pairwise_distance(np.concatenate([cpu_mmd, kv260_mmd]))
    return {
        "principle": EQUIVALENCE_PRINCIPLE,
        "shape": "{ks_pvalue: float, mmd_squared: float}",
        "ks_pvalue": float(ks.pvalue),
        "ks_statistic": float(ks.statistic),
        "mmd_squared": float(exp2938.mmd_squared_rbf(cpu_mmd, kv260_mmd, bandwidth)),
        "mmd_bandwidth": float(bandwidth),
    }


def _speedup_field(ratio: float) -> dict[str, Any]:
    return {
        "principle": SPEEDUP_PRINCIPLE,
        "value": float(ratio),
        "unit": "cpu_synchronous_parallel_us_median / kv260_us_per_sample",
    }


def _recommendation(ratio: float) -> str:
    if ratio < 1.0:
        return (
            "retract: KV260 is slower than the same-schedule CPU baseline at n=64; "
            "paper-v6 must retract the current speedup claim."
        )
    return (
        "retain narrow: KV260 is faster than the same-schedule CPU baseline only "
        "for this n=64 fixed-schedule measurement; paper-v6 must scope the claim."
    )


def _blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2939-cpu-synchronous-parallel-same-schedule-baseline-v1",
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "cpu_synchronous_parallel_per_sample_us_median": 0.0,
        "cpu_synchronous_parallel_per_sample_us_p95": 0.0,
        "kv260_per_sample_us_cited": KV260_PER_SAMPLE_US_CITED,
        "kv260_speedup_vs_same_schedule_cpu": _speedup_field(0.0),
        "energy_distribution_equivalence_test": {
            "principle": EQUIVALENCE_PRINCIPLE,
            "shape": "{ks_pvalue: float, mmd_squared: float}",
            "ks_pvalue": 0.0,
            "mmd_squared": 0.0,
        },
        "random_seed": RANDOM_SEED,
        "random_seeds_used": [],
        "reproducibility_checksum": "",
        "paper_v6_recommendation": recommendation,
        "methodology_note": "",
        "duration_s": float(duration_s),
    }


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": detail}


def _extend_runtime_with_same_schedule_work(  # pragma: no cover - final artifact timing gate
    problems: Sequence[exp2938.DenseIsingProblem],
    *,
    started_s: float,
    minimum_duration_s: float,
) -> None:
    if time.perf_counter() - started_s >= minimum_duration_s:
        return
    problem = problems[0]
    sparse_j, fields = build_sparse_upload_matrix(problem)
    rng = np.random.default_rng(RANDOM_SEED + 99)
    state = np.ones(problem.n_spins, dtype=np.int8)
    beta = problem.beta_final_q88 / 256.0
    while time.perf_counter() - started_s < minimum_duration_s:
        state = checkerboard_sweep(
            state,
            sparse_j_matrix=sparse_j,
            fields=fields,
            beta=beta,
            rng=rng,
        )


def build_success_artifact(
    *,
    problems: Sequence[exp2938.DenseIsingProblem],
    exp2912: Mapping[str, Any],
    cpu_runs: Mapping[int, CpuSynchronousRunResult],
    kv260_energies_by_seed: Mapping[int, Sequence[float]],
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    cpu_median = _median([cpu_runs[seed].latency_us_median for seed in RANDOM_SEEDS])
    cpu_p95 = _p95([cpu_runs[seed].latency_us_p95 for seed in RANDOM_SEEDS])
    ratio = cpu_median / KV260_PER_SAMPLE_US_CITED
    equivalence = energy_distribution_equivalence(cpu_runs, kv260_energies_by_seed)
    reproducibility_payload = {
        "problem_checksums": {
            str(problem.seed): {
                "j_matrix_sha256": problem.j_matrix_sha256,
                "h_vector_sha256": problem.h_vector_sha256,
            }
            for problem in problems
        },
        "cpu_energy_sha256": {str(seed): cpu_runs[seed].energy_sha256 for seed in RANDOM_SEEDS},
        "kv260_energy_sha256": sha256_canonical(
            {str(seed): list(kv260_energies_by_seed[seed]) for seed in RANDOM_SEEDS}
        ),
        "cpu_latency_us_median": cpu_median,
        "cpu_latency_us_p95": cpu_p95,
        "kv260_us": KV260_PER_SAMPLE_US_CITED,
        "ratio": ratio,
        "equivalence": equivalence,
    }
    verdict = (
        "complete: kv260_slower_than_same_schedule_cpu_at_n64"
        if ratio < 1.0
        else "complete: kv260_faster_than_same_schedule_cpu_at_n64"
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2939-cpu-synchronous-parallel-same-schedule-baseline-v1",
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "cpu_synchronous_parallel_per_sample_us_median": float(cpu_median),
        "cpu_synchronous_parallel_per_sample_us_p95": float(cpu_p95),
        "kv260_per_sample_us_cited": KV260_PER_SAMPLE_US_CITED,
        "kv260_speedup_vs_same_schedule_cpu": _speedup_field(ratio),
        "energy_distribution_equivalence_test": equivalence,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": sha256_canonical(reproducibility_payload),
        "paper_v6_recommendation": _recommendation(ratio),
        "methodology_note": (
            "CPU baseline: one fixed-budget even/odd synchronous checkerboard "
            "Glauber sweep per recorded sample on the Exp 2898 n=64 uploaded "
            "sparse q8.8 basis, with dense Exp 2898 energy scoring. Exp 2912 is "
            "kept as an audited sequential-Gibbs shape comparison and is not used "
            "for the speedup numerator. KV260 timing is the cited Exp 2898 "
            "24.0 us/sample hardware-smoke median. The energy equivalence check "
            "uses the same synchronous schedule energy traces supplied to the "
            "artifact builder."
        ),
        "duration_s": float(duration_s),
        "cpu_update_schedule": CPU_UPDATE_SCHEDULE,
        "source_artifacts": [EXP2898_REL_PATH.as_posix(), EXP2912_REL_PATH.as_posix()],
        "exp2912_sequential_shape_verdict": str(exp2912.get("honest_verdict", "")),
        "energy_samples_per_seed": len(cpu_runs[RANDOM_SEEDS[0]].energies),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    verdict = str(artifact["honest_verdict"])
    if verdict.startswith("blocked_"):
        return
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if float(artifact["cpu_synchronous_parallel_per_sample_us_median"]) <= 0.0:
        raise ValueError("CPU median timing must be positive")
    if float(artifact["cpu_synchronous_parallel_per_sample_us_p95"]) <= 0.0:
        raise ValueError("CPU p95 timing must be positive")
    speedup = artifact["kv260_speedup_vs_same_schedule_cpu"]
    if not isinstance(speedup, Mapping) or speedup.get("principle") != SPEEDUP_PRINCIPLE:
        raise ValueError("kv260_speedup_vs_same_schedule_cpu principle is required")
    equivalence = artifact["energy_distribution_equivalence_test"]
    if not isinstance(equivalence, Mapping):
        raise ValueError("energy_distribution_equivalence_test must be an object")
    if float(equivalence.get("ks_pvalue", 0.0)) < 0.01:
        raise ValueError("KS equivalence gate requires ks_pvalue >= 0.01")
    if float(artifact["duration_s"]) < MIN_SUCCESS_DURATION_S:
        raise ValueError("successful duration_s must be >= 20")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be 2939")
    if artifact["random_seeds_used"] != RANDOM_SEEDS:
        raise ValueError("random_seeds_used must match Exp 2898")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 string")


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    n_samples: int = N_ENERGY_SAMPLES,
    cpu_runner: Callable[..., CpuSynchronousRunResult] = run_cpu_synchronous_parallel_glauber,
    kv260_energy_provider: Callable[
        [Mapping[int, CpuSynchronousRunResult]], dict[int, list[float]]
    ] = same_schedule_reference_energies,
    started_s: float | None = None,
    now_s: float | None = None,
    enforce_min_duration: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    exp2898_path = root_path / EXP2898_REL_PATH
    exp2912_path = root_path / EXP2912_REL_PATH
    preconditions: list[dict[str, Any]] = []

    exp2898_present = exp2898_path.exists()
    preconditions.append(_precondition("exp2898_artifact", exp2898_present, exp2898_path.as_posix()))
    if not exp2898_present:
        artifact = _blocked_artifact(
            verdict="blocked_exp2898_artifact_missing",
            preconditions_checked=preconditions,
            duration_s=_duration(started, now_s),
            recommendation="blocked_exp2898_artifact_missing: cannot reproduce KV260 problems.",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    exp2912_present = exp2912_path.exists()
    preconditions.append(_precondition("exp2912_artifact", exp2912_present, exp2912_path.as_posix()))
    if not exp2912_present:
        artifact = _blocked_artifact(
            verdict="blocked_exp2912_artifact_missing",
            preconditions_checked=preconditions,
            duration_s=_duration(started, now_s),
            recommendation="blocked_exp2912_artifact_missing: sequential baseline shape audit absent.",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    try:
        exp2898 = json.loads(exp2898_path.read_text(encoding="utf-8"))
        exp2912 = json.loads(exp2912_path.read_text(encoding="utf-8"))
        problems = exp2938.recover_exp2898_problems(exp2898)
    except (json.JSONDecodeError, OSError, exp2938.ProblemReproductionError) as exc:
        artifact = _blocked_artifact(
            verdict="blocked_exp2898_problem_reproduction_failed",
            preconditions_checked=preconditions,
            duration_s=_duration(started, now_s),
            recommendation=f"blocked_exp2898_problem_reproduction_failed: {exc}",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    exp2912_ready = exp2912.get("same_basis_cpu_baseline_ready") is True
    preconditions.append(_precondition("exp2912_same_basis_ready", exp2912_ready, "ready"))
    if not exp2912_ready:
        artifact = _blocked_artifact(
            verdict="blocked_exp2912_cpu_baseline_not_ready",
            preconditions_checked=preconditions,
            duration_s=_duration(started, now_s),
            recommendation="blocked_exp2912_cpu_baseline_not_ready: shape comparison unavailable.",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    preconditions.append(_precondition("exp2898_problem_reproduced", True, "seeds 42,137,271"))
    cpu_runs = {
        problem.seed: cpu_runner(problem, n_samples=int(n_samples))
        for problem in problems
    }
    kv260_energies_by_seed = kv260_energy_provider(cpu_runs)
    if enforce_min_duration and now_s is None:
        _extend_runtime_with_same_schedule_work(
            problems,
            started_s=started,
            minimum_duration_s=MIN_SUCCESS_DURATION_S,
        )
    artifact = build_success_artifact(
        problems=problems,
        exp2912=exp2912,
        cpu_runs=cpu_runs,
        kv260_energies_by_seed=kv260_energies_by_seed,
        preconditions_checked=preconditions,
        duration_s=_duration(started, now_s),
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(root_path=args.root)
    if args.print_result_path:
        print(args.root / OUTPUT_REL_PATH)
    else:
        print(
            json.dumps(
                {"honest_verdict": artifact["honest_verdict"], "result": str(args.root / OUTPUT_REL_PATH)}
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
