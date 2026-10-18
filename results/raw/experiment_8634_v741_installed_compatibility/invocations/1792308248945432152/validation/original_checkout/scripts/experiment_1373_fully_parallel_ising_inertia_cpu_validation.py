#!/usr/bin/env python3
"""Exp 1373: fully parallel Ising inertia CPU validation on FoVer constraints.

**Researcher summary:**
    arXiv:2604.17109 shows that adding inertia to p-bit Ising dynamics can make
    fully synchronous spin updates stable enough for dense FPGA implementations.
    Carnot's production JAX sampler currently defaults to checkerboard updates,
    so this experiment performs a CPU-only validation of the same direction:
    run checked-in FoVer-derived constraint problems with checkerboard Gibbs and
    compare them to the new fully synchronous EMA-inertia update.

**Important scope boundary:**
    This script runs only on CPU/JAX. It estimates KV260 v4 RTL area from the
    existing checked-in design notes; it does not synthesize RTL, flash a board,
    run Vivado, or execute on KV260 hardware.

Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402

from carnot.samplers.parallel_ising import (  # noqa: E402
    _checkerboard_update,
    _inertia_parallel_update,
)


RUN_DATE = "20260505"
ALPHAS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1373_fully_parallel_ising_inertia_cpu_validation.json"
DEFAULT_FOVER_PATH = REPO_ROOT / "data/fover_corpus.jsonl"


@dataclass(frozen=True)
class FoVerConstraintProblem:
    """Dense Ising problem deterministically derived from one FoVer corpus row.

    Why this adapter exists: FoVer rows are natural-language/math verification
    examples, not pre-packaged Ising matrices. For this CPU validation we need a
    reproducible dense constraint graph whose identity is tied to real checked-in
    FoVer rows. The row text and label seed a target assignment, dense couplings,
    and biases; the resulting graph is a small benchmark proxy, not a new FoVer
    labeler.
    """

    name: str
    question_id: str
    label: str
    n_spins: int
    biases: np.ndarray
    coupling_matrix: np.ndarray
    ground_energy: float


@dataclass(frozen=True)
class TrialResult:
    """Single sampler trial summary used for aggregate convergence metrics."""

    steps_to_convergence: int
    final_energy: float
    stable_no_period2: bool


def _stable_name_part(raw: object) -> str:
    """Return a filesystem/JSON-safe short identifier for a corpus field."""
    text = str(raw) if raw is not None else "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "unknown"


def _load_fover_rows(path: Path, n_rows: int) -> list[dict]:
    """Load the first ``n_rows`` checked-in FoVer JSONL rows."""
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n_rows:
                break
    if len(rows) < n_rows:
        raise ValueError(f"needed {n_rows} FoVer rows from {path}, found {len(rows)}")
    return rows


def _digest_bytes(text: str) -> bytes:
    """Hash row content into deterministic bytes for reproducible graph features."""
    return hashlib.sha256(text.encode("utf-8")).digest()


def _problem_from_fover_row(
    row: dict,
    row_index: int,
    n_spins: int = 12,
) -> FoVerConstraintProblem:
    """Create a dense Ising constraint problem from one FoVer row.

    The target assignment is seeded by row text and label. Biases encourage the
    target bits directly, while a dense symmetric coupling matrix adds pairwise
    consistency pressure plus small deterministic row-specific perturbations.
    The graph is deliberately dense because the PIMI paper targets the failure
    mode where synchronous updates on dense graphs oscillate.
    """
    row_text = str(row.get("step_text") or row.get("response") or row.get("question") or "")
    label = str(row.get("label") or row.get("is_correct") or "unknown")
    question_id = _stable_name_part(row.get("question_id") or row.get("question_index") or row_index)
    seed_material = f"{question_id}|{label}|{row_text}"
    digest = _digest_bytes(seed_material)
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)

    target_bits = np.unpackbits(np.frombuffer(_digest_bytes(seed_material + "|target"), dtype=np.uint8))
    target = target_bits[:n_spins].astype(np.float64)
    target_pm = 2.0 * target - 1.0

    # Strong enough to make a well-defined constraint optimum, but not so strong
    # that either sampler trivially converges on every seed in one sweep.
    biases = 0.65 * target_pm
    perturb = rng.normal(loc=0.0, scale=0.08, size=(n_spins, n_spins))
    perturb = (perturb + perturb.T) / 2.0
    pair_constraints = 0.24 * np.outer(target_pm, target_pm)
    J = (pair_constraints + perturb) / np.sqrt(n_spins)
    np.fill_diagonal(J, 0.0)

    ground_energy = _bruteforce_ground_energy(biases, J)
    return FoVerConstraintProblem(
        name=f"fover_{question_id}_row{row_index}",
        question_id=question_id,
        label=label,
        n_spins=n_spins,
        biases=biases.astype(np.float32),
        coupling_matrix=J.astype(np.float32),
        ground_energy=ground_energy,
    )


def _bruteforce_ground_energy(biases: np.ndarray, coupling_matrix: np.ndarray) -> float:
    """Return the exact minimum boolean-Ising energy for small benchmark graphs."""
    n_spins = int(biases.shape[0])
    states = ((np.arange(2**n_spins)[:, None] >> np.arange(n_spins)) & 1).astype(np.float64)
    linear = states @ biases.astype(np.float64)
    pair = 0.5 * np.einsum("bi,ij,bj->b", states, coupling_matrix.astype(np.float64), states)
    energies = -(linear + pair)
    return float(np.min(energies))


def _energy(spins: np.ndarray, biases: np.ndarray, coupling_matrix: np.ndarray) -> float:
    """Compute the boolean convention energy used by ``parallel_ising.py``."""
    s = np.asarray(spins, dtype=np.float64)
    b = np.asarray(biases, dtype=np.float64)
    J = np.asarray(coupling_matrix, dtype=np.float64)
    return float(-(b @ s + 0.5 * s @ J @ s))


def _near_ground_threshold(ground_energy: float) -> float:
    """Relax exact optimum slightly so stochastic chains are not over-penalized."""
    return ground_energy + max(0.05 * abs(ground_energy), 1e-6)


def _has_period2_tail(states: list[np.ndarray]) -> bool:
    """Detect a simple exact period-2 oscillation in the last sampled states."""
    if len(states) < 6:
        return False
    tail = states[-6:]
    first_phase = all(np.array_equal(tail[i], tail[i - 2]) for i in range(2, 6))
    second_phase_differs = not np.array_equal(tail[-1], tail[-2])
    return bool(first_phase and second_phase_differs)


def _checkerboard_trial(
    problem: FoVerConstraintProblem,
    seed: int,
    max_sweeps: int,
    beta: float,
) -> TrialResult:
    """Run the existing checkerboard update loop until near-ground convergence."""
    rng = np.random.default_rng(seed)
    init = (rng.random(problem.n_spins) < 0.5).astype(np.float32)
    spins = jnp.asarray(init, dtype=jnp.float32)
    b = jnp.asarray(problem.biases, dtype=jnp.float32)
    J = jnp.asarray(problem.coupling_matrix, dtype=jnp.float32)
    even_mask = jnp.arange(problem.n_spins) % 2 == 0
    odd_mask = ~even_mask
    threshold = _near_ground_threshold(problem.ground_energy)
    states_tail: list[np.ndarray] = []
    key = jrandom.PRNGKey(seed + 10_000)
    final_energy = _energy(init, problem.biases, problem.coupling_matrix)

    for step in range(1, max_sweeps + 1):
        key, key_even, key_odd = jrandom.split(key, 3)
        spins = _checkerboard_update(
            spins,
            b,
            J,
            jnp.float32(beta),
            key_even,
            key_odd,
            even_mask,
            odd_mask,
        )
        state_np = np.asarray(spins)
        states_tail.append(state_np.copy())
        states_tail = states_tail[-6:]
        final_energy = _energy(state_np, problem.biases, problem.coupling_matrix)
        if final_energy <= threshold:
            return TrialResult(step, final_energy, not _has_period2_tail(states_tail))

    return TrialResult(max_sweeps, final_energy, not _has_period2_tail(states_tail))


def _inertia_trial(
    problem: FoVerConstraintProblem,
    seed: int,
    max_sweeps: int,
    beta: float,
    alpha: float,
) -> TrialResult:
    """Run fully synchronous EMA-inertia updates until near-ground convergence."""
    rng = np.random.default_rng(seed)
    init = (rng.random(problem.n_spins) < 0.5).astype(np.float32)
    spins = jnp.asarray(init, dtype=jnp.float32)
    b = jnp.asarray(problem.biases, dtype=jnp.float32)
    J = jnp.asarray(problem.coupling_matrix, dtype=jnp.float32)
    field_ema = jnp.zeros(problem.n_spins, dtype=jnp.float32)
    threshold = _near_ground_threshold(problem.ground_energy)
    states_tail: list[np.ndarray] = []
    key = jrandom.PRNGKey(seed + 20_000)
    final_energy = _energy(init, problem.biases, problem.coupling_matrix)

    for step in range(1, max_sweeps + 1):
        key, subkey = jrandom.split(key)
        spins, field_ema = _inertia_parallel_update(
            spins,
            b,
            J,
            jnp.float32(beta),
            subkey,
            field_ema,
            jnp.float32(alpha),
        )
        state_np = np.asarray(spins)
        states_tail.append(state_np.copy())
        states_tail = states_tail[-6:]
        final_energy = _energy(state_np, problem.biases, problem.coupling_matrix)
        if final_energy <= threshold:
            return TrialResult(step, final_energy, not _has_period2_tail(states_tail))

    return TrialResult(max_sweeps, final_energy, not _has_period2_tail(states_tail))


def _mean(values: Iterable[float]) -> float:
    """Return a JSON-friendly mean."""
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def _mapping_estimate() -> dict:
    """Return the KV260 v4 RTL LUT estimate without claiming synthesis."""
    return {
        "target": "KV260 v4 RTL estimate for N=128 fully synchronous inertia path",
        "n_spins": 128,
        "kv260_sparse_neighbors": 16,
        "dense_pimi_n128_lut_estimate": 290000,
        "kv260_v4_sparse_lut_estimate": 35872,
        "xck26_lut_budget": 117120,
        "fit_assessment": "dense fully unrolled PIMI does not fit KV260; sparse K=16 v4 estimate fits",
        "estimate_basis": [
            "hardware/kv260/ising_sampler_v4_spec.md LUT breakdown",
            "research-hardware-wishlist.md KV260 v4 sparse-inertia plan",
            "arXiv:2604.17109 PIMI fully synchronous update architecture; no local synthesis",
        ],
        "synthesis_performed": False,
        "board_executed": False,
    }


def run_experiment(
    output_path: Path = DEFAULT_RESULT_PATH,
    n_problems: int = 5,
    n_spins: int = 12,
    max_sweeps: int = 64,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    alphas: tuple[float, ...] = ALPHAS,
    beta: float = 2.0,
    fover_path: Path = DEFAULT_FOVER_PATH,
) -> dict:
    """Run the CPU validation and write the result artifact."""
    rows = _load_fover_rows(fover_path, n_problems)
    problems = [
        _problem_from_fover_row(row, row_index=i, n_spins=n_spins) for i, row in enumerate(rows)
    ]

    baseline_by_problem: list[dict] = []
    baseline_steps_all: list[int] = []
    for problem in problems:
        trials = [_checkerboard_trial(problem, seed, max_sweeps, beta) for seed in seeds]
        steps = [trial.steps_to_convergence for trial in trials]
        baseline_steps_all.extend(steps)
        baseline_by_problem.append(
            {
                "problem": problem.name,
                "mean_sweeps": _mean(steps),
                "ground_energy": problem.ground_energy,
            }
        )

    alpha_summaries: list[dict] = []
    inertia_by_alpha: dict[float, list[TrialResult]] = {}
    for alpha in alphas:
        trials_for_alpha: list[TrialResult] = []
        per_problem = []
        for problem in problems:
            trials = [
                _inertia_trial(problem, seed, max_sweeps, beta, alpha) for seed in seeds
            ]
            trials_for_alpha.extend(trials)
            per_problem.append(
                {
                    "problem": problem.name,
                    "mean_sweeps": _mean(trial.steps_to_convergence for trial in trials),
                    "stability": _mean(1.0 if trial.stable_no_period2 else 0.0 for trial in trials),
                }
            )
        inertia_by_alpha[float(alpha)] = trials_for_alpha
        mean_steps = _mean(trial.steps_to_convergence for trial in trials_for_alpha)
        alpha_summaries.append(
            {
                "alpha": float(alpha),
                "mean_sweeps": mean_steps,
                "speedup_vs_checkerboard": _mean(baseline_steps_all) / mean_steps
                if mean_steps > 0
                else None,
                "parallel_update_stability": _mean(
                    1.0 if trial.stable_no_period2 else 0.0 for trial in trials_for_alpha
                ),
                "per_problem": per_problem,
            }
        )

    best_summary = min(alpha_summaries, key=lambda item: item["mean_sweeps"])
    best_alpha = float(best_summary["alpha"])
    baseline_mean = _mean(baseline_steps_all)
    best_inertia_mean = float(best_summary["mean_sweeps"])
    speedup = baseline_mean / best_inertia_mean if best_inertia_mean > 0 else None
    best_trials = inertia_by_alpha[best_alpha]
    stability = _mean(1.0 if trial.stable_no_period2 else 0.0 for trial in best_trials)

    if best_alpha > 0.0 and speedup is not None and speedup > 1.0:
        verdict = "cpu_only_inertia_speedup_observed_no_hardware_claim"
    elif speedup is not None and speedup > 1.0:
        verdict = "cpu_only_fully_parallel_no_inertia_best_no_hardware_claim"
    else:
        verdict = "cpu_only_inertia_no_checkerboard_speedup_no_hardware_claim"

    artifact = {
        "status": "complete",
        "constraint_problems_tested": [problem.name for problem in problems],
        "inertia_alpha_sweep": alpha_summaries,
        "best_inertia_alpha": best_alpha,
        "inertia_convergence_speedup": speedup,
        "parallel_update_stability": stability,
        "steps_to_convergence_baseline": {
            "mean_sweeps": baseline_mean,
            "per_problem": baseline_by_problem,
            "sampler": "ParallelIsingSampler checkerboard update loop",
        },
        "steps_to_convergence_inertia": {
            "best_alpha": best_alpha,
            "mean_sweeps": best_inertia_mean,
            "per_alpha": alpha_summaries,
            "sampler": "InertiaIsingSampler fully synchronous EMA update loop",
        },
        "fpga_mapping_estimate": _mapping_estimate(),
        "hardware_claim_allowed": False,
        "kv260_claim_allowed": False,
        "honest_verdict": verdict,
        "metadata": {
            "experiment_id": 1373,
            "run_date": RUN_DATE,
            "project_root": str(REPO_ROOT),
            "fover_source": str(fover_path),
            "cpu_only": True,
            "n_spins_per_problem": n_spins,
            "max_sweeps": max_sweeps,
            "seeds": list(seeds),
            "beta": beta,
            "arxiv_reference": "https://arxiv.org/abs/2604.17109",
            "arxiv_update_rule_identified": (
                "s_i(t+1)=sign(tanh(beta(t) I_i(t)) + xi s_i(t) + eta(t) N(0,1)); "
                "Carnot CPU validation uses EMA-smoothed local field in the Bernoulli probability path."
            ),
            "problem_summaries": [
                {
                    "name": problem.name,
                    "question_id": problem.question_id,
                    "label": problem.label,
                    "n_spins": problem.n_spins,
                    "ground_energy": problem.ground_energy,
                }
                for problem in problems
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> None:
    """CLI entry point."""
    artifact = run_experiment()
    print(
        artifact["best_inertia_alpha"],
        artifact["inertia_convergence_speedup"],
        artifact["hardware_claim_allowed"],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":
    main()
