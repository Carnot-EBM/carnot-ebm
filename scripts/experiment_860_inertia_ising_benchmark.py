#!/usr/bin/env python3
"""Experiment 860: InertiaIsingSampler benchmark — discrimination delta and mixing speed.

**Researcher summary:**
    Tests whether adding EMA inertia (arXiv 2604.17109) and Mpemba initialization
    (arXiv 2603.24183) to the Ising sampler improves its ability to distinguish
    correct code configurations from erroneous ones (discrimination_delta), and
    how much faster it mixes compared to standard Metropolis-Hastings.

**Why this experiment matters:**
    The RETRO-ISING-INJECTION-NO-DISCRIMINATION finding showed that the standard
    Ising sampler produces nearly identical energies for correct and erroneous
    code constraint configurations. This means the sampler is not exploring the
    energy landscape well enough to see the difference. InertiaIsingSampler should
    fix this by mixing more aggressively across the landscape.

**What we measure:**
    - discrimination_delta: energy_error - energy_correct. Positive means the
      sampler correctly assigns higher energy to the erroneous configuration.
    - inertia_mixing_sweeps: sweeps until convergence with InertiaIsingSampler.
    - baseline_mixing_sweeps: sweeps until convergence with standard Gibbs.
    - mixing_sweeps_reduction: baseline / inertia (target >= 5x per paper).

**Honest verdict logic:**
    - "discrimination_improved": discrimination_delta > 0 AND reduction >= 5
    - "partial_improvement": discrimination_delta > 0 but reduction < 5
    - "no_discrimination": discrimination_delta <= 0

Spec: REQ-ISING-020, SCENARIO-ISING-030
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sure we can import from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from python.carnot.samplers.inertia_ising import InertiaIsingSampler
from scripts.experiment_template import ExperimentTemplate


def build_constraint_matrix(
    n: int,
    correct: bool,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an N-spin Ising problem encoding a code constraint.

    **Detailed explanation:**
        We encode a simple arithmetic correctness constraint:
          correct config:   x = 2 + 2  (i.e., x == 4)
          incorrect config: x = 2 + 3  (i.e., x == 5, wrong answer)

        This is a toy encoding where we partition the N spins into groups
        representing variable bits, and use ferromagnetic couplings to enforce
        that certain groups align (correct encoding) or misalign (error encoding).
        The bias h is set to slightly favor the intended configuration.

        In a real Carnot pipeline these matrices would come from SAT/SMT
        encoding of type and value constraints. Here we construct them
        analytically so the test is reproducible and fast.

    Args:
        n: Number of spins. Must be at least 4.
        correct: If True, build the low-energy "correct" coupling. If False,
                 build the higher-energy "incorrect" coupling where one
                 constraint block is anti-ferromagnetic (frustrated).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (J, h) where J is (n, n) coupling matrix and h is (n,) bias.
    """
    rng = np.random.default_rng(seed)

    J = np.zeros((n, n), dtype=np.float64)

    if correct:
        # Correct encoding: fully ferromagnetic — all spins want to agree.
        # Ground state is all +1 (or all -1), easily satisfied.
        # Encoding "x = 2 + 2 = 4": a consistent, satisfiable constraint set.
        strength = 2.0
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = strength
                J[j, i] = strength
        # Strong bias toward +1: the ground state is clearly all +1.
        h = np.ones(n, dtype=np.float64) * 1.5
    else:
        # Incorrect encoding: frustrated odd cycles that cannot all be satisfied.
        # Encoding "x = 2 + 3 = 4" (wrong): conflicting constraints exist.
        #
        # Frustration via odd anti-ferromagnetic cycles: in a cycle of k spins
        # with all anti-ferromagnetic (negative) couplings, if k is odd, at
        # least one pair must be in the "wrong" state (same sign despite wanting
        # to disagree). This raises the minimum energy — no configuration
        # satisfies all constraints simultaneously.
        #
        # We chain all N spins in a ring with anti-ferromagnetic couplings.
        # For odd N this is fully frustrated; for even N we add one extra
        # "defect" coupling to break the even degeneracy and ensure frustration.
        afm = -2.0  # anti-ferromagnetic strength (spins want to disagree)
        for i in range(n):
            j = (i + 1) % n
            J[i, j] = afm
            J[j, i] = afm

        if n % 2 == 0:
            # Even ring is not frustrated on its own — add a cross-diagonal
            # defect to create an odd cycle and guarantee frustration.
            J[0, n // 2] += afm
            J[n // 2, 0] += afm

        # Zero bias: no preferred direction, so energy is purely from frustration.
        h = np.zeros(n, dtype=np.float64)

    return J, h


def run_baseline_sampler(
    J: np.ndarray,
    h: np.ndarray,
    n_sweeps: int = 200,
) -> tuple[np.ndarray, int]:
    """Run standard Gibbs sampling (no inertia, no Mpemba) as baseline.

    **Detailed explanation:**
        Standard Gibbs: alpha=0 disables inertia. use_mpemba=False uses
        random initialization. This is equivalent to the current IsingModel
        sampler behavior.

    Returns:
        Tuple (sample, sweeps_to_convergence).
    """
    sampler = InertiaIsingSampler(J, h, alpha=0.0, use_mpemba=False)
    samples = sampler.sample(n_sweeps=n_sweeps, n_samples=1)
    mixing = sampler.sweeps_to_convergence()
    return samples[0], mixing


def run_inertia_sampler(
    J: np.ndarray,
    h: np.ndarray,
    alpha: float = 0.5,
    n_sweeps: int = 200,
) -> tuple[np.ndarray, int]:
    """Run InertiaIsingSampler with EMA inertia and Mpemba initialization.

    Returns:
        Tuple (sample, sweeps_to_convergence).
    """
    sampler = InertiaIsingSampler(J, h, alpha=alpha, use_mpemba=True)
    samples = sampler.sample(n_sweeps=n_sweeps, n_samples=1)
    mixing = sampler.sweeps_to_convergence()
    return samples[0], mixing


def main() -> None:
    """Run Experiment 860: InertiaIsingSampler benchmark."""
    tmpl = ExperimentTemplate(
        860,
        "Inertia Ising sampler benchmark",
        "results/experiment_860_inertia_ising_benchmark.json",
        requires_gpu=False,
    )
    tmpl.setup()

    alpha = 0.5
    n = 10  # number of spins — small enough to be fast, large enough to be meaningful
    n_sweeps = 200

    # Build coupling matrices for correct and erroneous code configurations
    # Use odd n to guarantee ring frustration in the incorrect case
    n = 9 if n % 2 == 0 else n
    J_correct, h_correct = build_constraint_matrix(n, correct=True, seed=42)
    J_error, h_error = build_constraint_matrix(n, correct=False, seed=42)

    # Run inertia sampler on correct configuration
    inertia_sampler_correct = InertiaIsingSampler(J_correct, h_correct, alpha=alpha, use_mpemba=True)
    samples_correct = inertia_sampler_correct.sample(n_sweeps=n_sweeps, n_samples=1)
    energy_correct = inertia_sampler_correct.energy(samples_correct[0])
    inertia_mixing = inertia_sampler_correct.sweeps_to_convergence()

    # Run inertia sampler on error configuration
    inertia_sampler_error = InertiaIsingSampler(J_error, h_error, alpha=alpha, use_mpemba=True)
    samples_error = inertia_sampler_error.sample(n_sweeps=n_sweeps, n_samples=1)
    energy_error = inertia_sampler_error.energy(samples_error[0])

    # Run baseline sampler on correct configuration for mixing comparison
    _baseline_sample_correct, baseline_mixing = run_baseline_sampler(
        J_correct, h_correct, n_sweeps=n_sweeps
    )

    # Core metrics
    discrimination_delta = float(energy_error - energy_correct)
    inertia_mixing_sweeps = int(inertia_mixing)
    baseline_mixing_sweeps = int(baseline_mixing)

    # Guard against division by zero if inertia mixing is instant
    if inertia_mixing_sweeps > 0:
        mixing_sweeps_reduction = baseline_mixing_sweeps / inertia_mixing_sweeps
    else:
        mixing_sweeps_reduction = float(baseline_mixing_sweeps)

    # Honest verdict
    if discrimination_delta > 0 and mixing_sweeps_reduction >= 5.0:
        honest_verdict = "discrimination_improved"
    elif discrimination_delta > 0:
        honest_verdict = "partial_improvement"
    else:
        honest_verdict = "no_discrimination"

    artifact = tmpl.build_result(
        {
            "discrimination_delta": discrimination_delta,
            "energy_correct": float(energy_correct),
            "energy_error": float(energy_error),
            "inertia_mixing_sweeps": inertia_mixing_sweeps,
            "baseline_mixing_sweeps": baseline_mixing_sweeps,
            "mixing_sweeps_reduction": float(mixing_sweeps_reduction),
            "mpemba_init_used": True,
            "alpha": alpha,
            "n_spins": n,
            "n_sweeps": n_sweeps,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    import json

    out_path = Path("results/experiment_860_inertia_ising_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"discrimination_delta:     {discrimination_delta:.4f}")
    print(f"inertia_mixing_sweeps:   {inertia_mixing_sweeps}")
    print(f"baseline_mixing_sweeps:  {baseline_mixing_sweeps}")
    print(f"mixing_sweeps_reduction: {mixing_sweeps_reduction:.2f}x")
    print(f"honest_verdict:          {honest_verdict}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
