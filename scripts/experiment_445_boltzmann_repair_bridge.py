#!/usr/bin/env python3
"""Experiment 445: BoltzmannRepairBridge — energy-guided LLM repair direction.

**Researcher summary:**
    Validates BoltzmannRepairBridge (arXiv 2601.17094 insight): project Ising
    ground-state spin configurations to LLM embedding space via a trained linear
    adapter, replacing the naive "ask LLM to fix error" repair step with an
    energy-guided alternative.

    Experimental design:
    1. Build a 16-variable IsingEBM with arithmetic constraint couplings.
    2. Train LinearSpinAdapter(spin_dim=16, embed_dim=128) on 50 synthetic pairs.
    3. Run evaluate_repair_quality(n_samples=100) to measure energy reduction rate.
    4. Compare BoltzmannRepairBridge vs random repair baseline.
    5. Write honest verdict based on observed repair_success_rate.

    Honest verdict:
      'repair_energy_positive'  if repair_success_rate > 0.60
      'repair_energy_marginal'  if 0.40 < rate <= 0.60
      'no_energy_reduction'     otherwise

    CPU-only. Always produces a result JSON.

Spec: REQ-REPAIR-014, REQ-REPAIR-015,
      SCENARIO-REPAIR-028, SCENARIO-REPAIR-029, SCENARIO-REPAIR-030
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Always apply env autofix first (detects ROCm/CUDA, injects JAX platform vars).
sys.path.insert(0, str(Path(__file__).parent.parent))
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

# Force CPU-only JAX for reproducibility (no GPU allocation for this experiment).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.ising import IsingConfig, IsingModel
from carnot.pipeline.boltzmann_repair import (
    BoltzmannRepairBridge,
    LinearSpinAdapter,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 445
EXP_TITLE = "BoltzmannRepairBridge: Ising ground-state to LLM repair direction"
RESULT_PATH = Path("results/experiment_445_boltzmann_repair_bridge.json")
TIMEOUT_MINUTES = 20
SPIN_DIM = 16
EMBED_DIM = 128
N_TRAIN_PAIRS = 50
N_EVAL_SAMPLES = 100
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Arithmetic constraint coupling matrix builder
# ---------------------------------------------------------------------------


def build_arithmetic_constraint_couplings(n_vars: int, key: jax.Array) -> tuple:
    """Build Ising model with arithmetic constraint structure.

    **What the coupling matrix encodes:**
        For arithmetic constraints like "x_0 + x_1 = x_2", we encode the
        constraint as: x_0 and x_1 positively coupled (must agree on carry),
        and each negatively coupled to x_2 (since their sum determines it).

        This is a simplified encoding — production use would map real
        arithmetic constraints via the sat_to_ising() function. Here we
        create synthetic structured couplings to validate the pipeline.

    Returns:
        (IsingModel, coupling_description) where coupling_description is a
        human-readable summary of the constraint structure.
    """
    import jax

    config = IsingConfig(input_dim=n_vars, coupling_init="xavier_uniform")
    model = IsingModel(config, key=key)

    # The default xavier_uniform initialisation already creates structured
    # couplings. For the experiment, we reinforce the structure by adding
    # arithmetic-constraint-inspired couplings:
    # - Adjacent pairs (i, i+1) are positively coupled (carry propagation).
    # - Pairs (i, i+2) are negatively coupled (cancellation in addition).
    J = model.coupling

    # Build structured coupling perturbation.
    key_perturb, _ = jrandom.split(key)
    structured = jnp.zeros_like(J)

    for i in range(n_vars - 1):
        # Positive coupling: adjacent variables tend to agree.
        structured = structured.at[i, i + 1].set(0.5)
        structured = structured.at[i + 1, i].set(0.5)

    for i in range(n_vars - 2):
        # Negative coupling: skip-one variables tend to differ.
        structured = structured.at[i, i + 2].set(-0.3)
        structured = structured.at[i + 2, i].set(-0.3)

    # Blend xavier init with structured coupling (70% xavier, 30% structured).
    model.coupling = 0.7 * J + 0.3 * structured

    description = (
        f"{n_vars}-variable Ising: adjacent pairs +0.35 coupling (carry propagation), "
        f"skip-one pairs -0.15 coupling (addition cancellation)"
    )
    return model, description


# ---------------------------------------------------------------------------
# Synthetic training data builder
# ---------------------------------------------------------------------------


def build_training_data(
    n_pairs: int, spin_dim: int, embed_dim: int, seed: int
) -> tuple:
    """Build synthetic (spin_config, target_embedding) training pairs.

    **What this simulates:**
        In production, target_embeddings come from successful LLM repairs:
        we extract the embedding of the repair token from the LLM's hidden
        states, paired with the Ising spin config that described the violated
        constraints.

        Here we use random unit vectors as target embeddings. This is sufficient
        to validate that the adapter trains without error and produces finite
        MSE — we're testing the training mechanics, not the embedding quality.

    Returns:
        (spin_configs, target_embeddings, description) where:
        - spin_configs: shape (n_pairs, spin_dim), values ±1
        - target_embeddings: shape (n_pairs, embed_dim), unit vectors
    """
    key = jrandom.PRNGKey(seed)
    k1, k2 = jrandom.split(key)

    # Spin configs: random ±1.
    spins_bool = jrandom.bernoulli(k1, 0.5, (n_pairs, spin_dim))
    spin_configs = 2.0 * spins_bool.astype(jnp.float32) - 1.0

    # Target embeddings: random unit vectors (simulating LLM repair embeddings).
    raw = jrandom.normal(k2, (n_pairs, embed_dim))
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    target_embeddings = raw / (norms + 1e-8)

    description = (
        f"{n_pairs} synthetic pairs: spin_configs=±1 random, "
        f"target_embeddings=random unit vectors in R^{embed_dim}"
    )
    return spin_configs, target_embeddings, description


# ---------------------------------------------------------------------------
# Random repair baseline
# ---------------------------------------------------------------------------


def evaluate_random_baseline(
    ising_model: IsingModel, n_samples: int, seed: int
) -> dict:
    """Measure energy reduction for RANDOM repair (no energy guidance).

    **What this computes:**
        For each of n_samples random constraint states, generate TWO random spin
        configurations. Compute the fraction where the second random config has
        lower energy than the first. This is the baseline: if random repair
        succeeds ~50% of the time (as expected for i.i.d. random spins), and
        BoltzmannRepairBridge succeeds significantly more often, the bridge adds value.

    Returns:
        Dict with same keys as evaluate_repair_quality() for direct comparison.
    """
    key = jrandom.PRNGKey(seed + 1000)
    dim = ising_model.input_dim

    reductions = []
    successes = []
    energies_after = []

    for i in range(n_samples):
        key, k1, k2 = jrandom.split(key, 3)

        # "Before" state: first random spin config.
        spins_before = 2.0 * jrandom.bernoulli(k1, 0.5, (dim,)).astype(jnp.float32) - 1.0
        e_before = float(ising_model.energy(spins_before))

        # "After" state: second INDEPENDENT random spin config (no annealing).
        spins_after = 2.0 * jrandom.bernoulli(k2, 0.5, (dim,)).astype(jnp.float32) - 1.0
        e_after = float(ising_model.energy(spins_after))

        reduction = e_before - e_after
        reductions.append(reduction)
        successes.append(e_after < e_before)
        energies_after.append(e_after)

    return {
        "mean_energy_reduction": float(sum(reductions) / n_samples),
        "repair_success_rate": float(sum(successes) / n_samples),
        "n_samples": n_samples,
        "min_energy_after": float(min(energies_after)),
        "max_energy_after": float(max(energies_after)),
    }


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------


def compute_honest_verdict(bridge_rate: float, baseline_rate: float) -> str:
    """Compute honest verdict based on bridge repair success rate.

    **Verdict logic:**
    - 'repair_energy_positive': bridge_rate > 0.60 (strongly energy-guided)
    - 'repair_energy_marginal': 0.40 < bridge_rate <= 0.60 (weak signal)
    - 'no_energy_reduction': bridge_rate <= 0.40 (no improvement over random)

    Note: baseline_rate ≈ 0.50 is expected for random repair. If bridge_rate
    significantly exceeds 0.60, the energy guidance is working.
    """
    if bridge_rate > 0.60:
        return "repair_energy_positive"
    elif bridge_rate > 0.40:
        return "repair_energy_marginal"
    else:
        return "no_energy_reduction"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Run the full BoltzmannRepairBridge experiment and return the result artifact.

    Steps:
    a. Build 16-variable IsingEBM with arithmetic constraint structure.
    b. Train LinearSpinAdapter(spin_dim=16, embed_dim=128) on synthetic pairs.
    c. Run evaluate_repair_quality(n_samples=100) via BoltzmannRepairBridge.
    d. Compare with random repair baseline.
    e. Build and return artifact.
    """
    import time

    t_start = time.time()
    key = jrandom.PRNGKey(RANDOM_SEED)

    # Step a: Build IsingEBM.
    _log.info("Building 16-variable IsingEBM with arithmetic constraint couplings...")
    key, model_key = jrandom.split(key)
    ising_model, coupling_description = build_arithmetic_constraint_couplings(SPIN_DIM, model_key)
    _log.info("IsingEBM ready. %s", coupling_description)

    # Step b: Train LinearSpinAdapter.
    _log.info("Building %d synthetic training pairs...", N_TRAIN_PAIRS)
    spin_configs, target_embeddings, data_description = build_training_data(
        N_TRAIN_PAIRS, SPIN_DIM, EMBED_DIM, seed=RANDOM_SEED
    )

    _log.info("Training LinearSpinAdapter(spin_dim=%d, embed_dim=%d)...", SPIN_DIM, EMBED_DIM)
    key, adapter_key = jrandom.split(key)
    adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM, key=adapter_key)
    final_mse = adapter.train(spin_configs, target_embeddings, n_epochs=50)
    _log.info("Adapter trained. Final MSE loss: %.6f", final_mse)

    # Step c: Evaluate BoltzmannRepairBridge.
    _log.info("Building BoltzmannRepairBridge...")
    bridge = BoltzmannRepairBridge(
        ising_model=ising_model,
        adapter=adapter,
        n_warmup=200,
        n_samples=10,
        steps_per_sample=10,
        beta_final=10.0,
    )

    _log.info("Running evaluate_repair_quality(n_samples=%d)...", N_EVAL_SAMPLES)
    bridge_metrics = bridge.evaluate_repair_quality(n_samples=N_EVAL_SAMPLES, seed=RANDOM_SEED)
    _log.info(
        "Bridge metrics: repair_success_rate=%.3f, mean_energy_reduction=%.4f",
        bridge_metrics["repair_success_rate"],
        bridge_metrics["mean_energy_reduction"],
    )

    # Step d: Compare with random repair baseline.
    _log.info("Computing random repair baseline...")
    baseline_metrics = evaluate_random_baseline(ising_model, N_EVAL_SAMPLES, seed=RANDOM_SEED)
    _log.info(
        "Baseline metrics: repair_success_rate=%.3f, mean_energy_reduction=%.4f",
        baseline_metrics["repair_success_rate"],
        baseline_metrics["mean_energy_reduction"],
    )

    bridge_rate = bridge_metrics["repair_success_rate"]
    baseline_rate = baseline_metrics["repair_success_rate"]
    honest_verdict = compute_honest_verdict(bridge_rate, baseline_rate)

    t_end = time.time()
    duration_s = t_end - t_start

    _log.info("Honest verdict: %s", honest_verdict)
    _log.info("Experiment completed in %.1f seconds", duration_s)

    # Step e: Build result artifact.
    artifact = {
        "schema": "carnot.boltzmann_repair.v1",
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "honest_verdict": honest_verdict,
        "bridge_metrics": bridge_metrics,
        "baseline_metrics": baseline_metrics,
        "bridge_vs_baseline": {
            "success_rate_delta": bridge_rate - baseline_rate,
            "bridge_rate": bridge_rate,
            "baseline_rate": baseline_rate,
        },
        "adapter_training": {
            "n_train_pairs": N_TRAIN_PAIRS,
            "n_epochs": 50,
            "final_mse": final_mse,
            "data_description": data_description,
        },
        "ising_model": {
            "spin_dim": SPIN_DIM,
            "coupling_description": coupling_description,
        },
        "bridge_config": {
            "spin_dim": SPIN_DIM,
            "embed_dim": EMBED_DIM,
            "n_warmup": 200,
            "n_samples_per_call": 10,
            "steps_per_sample": 10,
            "beta_final": 10.0,
        },
        "duration_s": duration_s,
        "cpu_only": True,
        "spec": [
            "REQ-REPAIR-014",
            "REQ-REPAIR-015",
            "SCENARIO-REPAIR-028",
            "SCENARIO-REPAIR-029",
            "SCENARIO-REPAIR-030",
        ],
        "references": [
            "Boltzmann-GPT arXiv:2601.17094",
            "ARM-EBM arXiv:2512.15605",
        ],
    }

    return artifact


def main() -> None:
    """Entry point: run experiment with watchdog and write result JSON."""
    import jax  # noqa: F401 — ensure JAX is initialised before watchdog starts

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(RESULT_PATH),
    ):
        artifact = run_experiment()

    _log.info("Writing result to %s", RESULT_PATH)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info("Done. honest_verdict=%s", artifact["honest_verdict"])
    _log.info(
        "bridge_rate=%.3f  baseline_rate=%.3f  delta=%.3f",
        artifact["bridge_vs_baseline"]["bridge_rate"],
        artifact["bridge_vs_baseline"]["baseline_rate"],
        artifact["bridge_vs_baseline"]["success_rate_delta"],
    )


if __name__ == "__main__":
    import jax

    main()
