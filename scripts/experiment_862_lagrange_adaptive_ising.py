#!/usr/bin/env python3
"""Exp 862: LagrangeAdaptive Ising FR-11 self-learning relay (arXiv 2501.04971).

**Researcher summary:**
    FR-11 (Autonomous Self-Learning Loop) requires at least one self-learning experiment
    per milestone.  This experiment implements Lagrange relaxation of Ising constraint
    weights as described in arXiv 2501.04971.  When a constraint is violated, its
    coupling weight lambda_k increases proportionally to the violation severity, reshaping
    the energy landscape toward satisfaction.

    The 5-session relay design:
      - Session 1: Run LagrangeAdaptiveIsingConstraints on 10 synthetic binary constraints,
        5 of which initially encourage wrong alignment (J matrix biased against constraint
        satisfaction).  Measure violation_rate_s1.
      - Sessions 2–5: Start from updated lambdas.  The adaptive updates should push
        violated constraint lambdas up, making violations more energetically expensive.
      - delta_s1_to_s5 = vr_s1 - vr_s5.  Positive = self-learning occurred.

**Honest verdict logic:**
    - "fr11_self_learning_confirmed": delta_s1_to_s5 > 0 (violation rate decreased).
    - "fr11_no_improvement": delta_s1_to_s5 <= 0 (no self-learning detected).
    Both are valid results — the FR-11 requirement is satisfied by running the experiment
    honestly; improvement would confirm the mechanism but absence is also informative.

Spec: REQ-FR11-020, SCENARIO-FR11-030
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add the repo root to sys.path so experiment_template is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

from carnot.samplers.lagrange_adaptive import LagrangeAdaptiveIsingConstraints  # noqa: E402

_DELIVERABLE = "results/experiment_862_lagrange_adaptive_ising.json"

# Number of spins in the synthetic Ising problem.
_N_SPINS = 10

# Number of binary "agree" constraints — each says two spins must align.
_N_CONSTRAINTS = 10

# For 5 constraints, we inject a negative bias into J_base to bias the sampler
# AGAINST satisfaction initially (violation rate in session 1 should be higher).
_N_ADVERSARIAL = 5

# Sweeps and samples per session.
_N_SWEEPS = 200
_N_SAMPLES = 20

# Number of relay sessions.
_N_SESSIONS = 5


def _build_synthetic_constraints(rng: np.random.Generator) -> list[dict]:
    """Build 10 binary agree/disagree constraints on 10 spins.

    **Design rationale:**
        We create 10 distinct pairs (i, j) where i != j, all with sign=+1
        (the sampler should prefer s_i == s_j for each pair).  The penalty is 1.0
        for all constraints so differences in lambda are the only adaptive variable.

        Note: constraints use a fixed pairing: (0,1), (1,2), ..., (9,0) — a ring.
        This creates a frustrated ring topology: an odd-length antiferromagnetic ring
        is classically frustrated, but since all our constraints use sign=+1 (ferromagnetic),
        the ground state has all spins equal and ALL constraints satisfied.

    Args:
        rng: Unused; kept for API consistency.  Constraints are deterministic.

    Returns:
        List of 10 constraint dicts.
    """
    constraints = []
    for k in range(_N_CONSTRAINTS):
        i = k % _N_SPINS
        j = (k + 1) % _N_SPINS
        constraints.append({"spins": [i, j], "sign": 1, "penalty": 1.0})
    return constraints


def _adversarial_J_base(constraints: list[dict]) -> np.ndarray:
    """Build an adversarial J_base that biases against the first N_ADVERSARIAL constraints.

    **Why adversarial seeding:**
        Without an adversarial bias, the InertiaIsingSampler may already satisfy all
        constraints in session 1 (the problem is too easy).  We add a negative coupling
        term for the first 5 constraints to bias the energy landscape AGAINST satisfaction.
        This ensures session 1 has a measurable violation rate, so the Lagrange update
        has something to correct over 5 sessions.

    Returns:
        np.ndarray of shape (n_spins, n_spins) — adversarial base coupling matrix.
    """
    J_base = np.zeros((_N_SPINS, _N_SPINS), dtype=np.float64)
    for k in range(_N_ADVERSARIAL):
        c = constraints[k]
        i, j = c["spins"]
        # Negative coupling: penalises the preferred alignment (s_i == s_j).
        J_base[i, j] -= 2.0
        J_base[j, i] -= 2.0
    return J_base


def run_relay(constraints: list[dict], J_base: np.ndarray) -> dict:
    """Run the 5-session Lagrange adaptive relay.

    **Session relay logic:**
        Each session:
          1. Build J = J_base + lambda-weighted constraint couplings.
          2. Sample n_samples spin configs with InertiaIsingSampler.
          3. Measure violation rate.
          4. Update lambdas: violated constraints get higher lambda.

        The net effect: over sessions, lambda for violated constraints grows,
        increasing the coupling strength and reducing violations.

    Args:
        constraints: List of 10 constraint dicts.
        J_base: Adversarial base coupling matrix (biases session 1 toward violations).

    Returns:
        dict with per-session results and overall delta_s1_to_s5.
    """
    solver = LagrangeAdaptiveIsingConstraints(
        n_spins=_N_SPINS,
        n_constraints=_N_CONSTRAINTS,
        lambda_init=1.0,
        lambda_lr=0.2,
    )
    # Inject the adversarial base coupling so initial sessions violate constraints.
    solver.J_base = J_base

    session_results: list[dict] = []
    lambda_trajectory: list[list[float]] = []

    for session_idx in range(_N_SESSIONS):
        result = solver.run_session(
            constraints=constraints,
            n_sweeps=_N_SWEEPS,
            n_samples=_N_SAMPLES,
        )
        session_results.append(
            {
                "session": session_idx + 1,
                "violation_rate": result["violation_rate"],
                "per_constraint_violation_rates": result["per_constraint_violation_rates"],
            }
        )
        lambda_trajectory.append(list(result["lambdas"]))

    violation_rates = [s["violation_rate"] for s in session_results]
    delta_s1_to_s5 = violation_rates[0] - violation_rates[-1]

    return {
        "session_results": session_results,
        "violation_rates": violation_rates,
        "delta_s1_to_s5": delta_s1_to_s5,
        "lambda_trajectory": lambda_trajectory,
        "fr11_self_learning_confirmed": delta_s1_to_s5 > 0,
    }


def main() -> None:
    """Entry point for Exp 862."""
    tmpl = ExperimentTemplate(
        862,
        "LagrangeAdaptive Ising FR-11 self-learning",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    rng = np.random.default_rng(seed=862)
    constraints = _build_synthetic_constraints(rng)
    J_base = _adversarial_J_base(constraints)

    relay_data = run_relay(constraints, J_base)

    honest_verdict = (
        "fr11_self_learning_confirmed"
        if relay_data["fr11_self_learning_confirmed"]
        else "fr11_no_improvement"
    )

    artifact = tmpl.build_result(
        {
            "violation_rates": relay_data["violation_rates"],
            "delta_s1_to_s5": relay_data["delta_s1_to_s5"],
            "lambda_trajectory": relay_data["lambda_trajectory"],
            "fr11_self_learning_confirmed": relay_data["fr11_self_learning_confirmed"],
            "session_results": relay_data["session_results"],
            "n_sessions": _N_SESSIONS,
            "n_spins": _N_SPINS,
            "n_constraints": _N_CONSTRAINTS,
            "n_adversarial": _N_ADVERSARIAL,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Exp 862 complete — honest_verdict={honest_verdict}")
    print(f"  violation_rates: {relay_data['violation_rates']}")
    print(f"  delta_s1_to_s5:  {relay_data['delta_s1_to_s5']:.4f}")
    print(f"  fr11_self_learning_confirmed: {relay_data['fr11_self_learning_confirmed']}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
