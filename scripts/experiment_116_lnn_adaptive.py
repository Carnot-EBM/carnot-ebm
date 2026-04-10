#!/usr/bin/env python3
"""Experiment 116: LNN Adaptive Constraint Model vs Static Ising for Agentic Reasoning.

**Researcher summary:**
    Compares a Liquid Time-Constant Network (LNN) constraint model against a static
    Ising model for detecting errors in synthetic 5-step reasoning chains. The LNN
    adapts its energy landscape after each step; the Ising model does not. We measure
    which model detects reasoning errors faster and more reliably.

**Motivation (Research-Program Goal #8):**
    Static Ising cannot adapt as an agent acts over time. When step 1 establishes a
    key fact, the Ising model doesn't use that to evaluate step 3's consistency. An
    LNN-based constraint model updates coupling strengths after each observation,
    creating a context-sensitive energy landscape. This experiment tests whether that
    adaptation actually helps with error detection.

**Experimental design:**
    - 20 synthetic 5-step reasoning chains (10 correct, 10 with errors at steps 2-4)
    - Each step is represented as a 16-dimensional synthetic embedding
    - Correct steps: embeddings sampled from a consistent Gaussian cluster
    - Error steps: embeddings that deviate significantly from the cluster pattern
    - Both models evaluate each step's energy; LNN adapts between steps, Ising does not

**Metrics:**
    1. Error detection rate: fraction of error chains where the erroneous step has
       the highest energy among all steps (for that chain)
    2. Energy gap: mean(energy at error step) - mean(energy at correct steps) per chain
    3. Energy evolution: per-step energy trajectories for LNN vs Ising
    4. Step-of-first-detection: which step index first shows elevated energy

**Output:** results/experiment_116_results.json

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_116_lnn_adaptive.py

REQ: REQ-CORE-001, REQ-CORE-002
SCENARIO: SCENARIO-CORE-001
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX may crash).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow importing carnot from the python/ directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.lnn_constraint import LNNConstraintConfig, LNNConstraintModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 16         # Embedding dimensionality for each reasoning step
HIDDEN_DIM = 8         # LNN hidden state size
N_CHAINS = 20          # Total number of reasoning chains (10 correct + 10 with errors)
N_STEPS = 5            # Steps per reasoning chain
SEED = 2024            # Master RNG seed for reproducibility

# Cluster statistics for generating synthetic embeddings
# "Correct" steps cluster around the origin with low variance
CORRECT_STEP_MEAN = 0.0
CORRECT_STEP_STD = 0.3

# "Erroneous" steps deviate from the cluster — different mean/variance
ERROR_STEP_MEAN = 2.5   # Shifted mean (inconsistent with earlier steps)
ERROR_STEP_STD = 0.5    # Higher variance (more uncertain/incoherent)


# ---------------------------------------------------------------------------
# Synthetic chain generation
# ---------------------------------------------------------------------------

def generate_correct_chain(key: jax.Array) -> list[jnp.ndarray]:
    """Generate a 5-step reasoning chain where all steps are consistent.

    Each step is a 16-dim embedding sampled from a tight Gaussian cluster.
    The cluster mean drifts slightly across steps to simulate a coherent
    reasoning trajectory (each step builds on the previous).

    Args:
        key: JAX PRNG key for reproducibility.

    Returns:
        List of 5 step embeddings, each shape (INPUT_DIM,).
    """
    steps = []
    for step_idx in range(N_STEPS):
        key, subkey = jrandom.split(key)
        # Slight drift in mean: reasoning progresses coherently
        drift = jnp.ones(INPUT_DIM) * (step_idx * 0.05)
        x = jrandom.normal(subkey, (INPUT_DIM,)) * CORRECT_STEP_STD + drift + CORRECT_STEP_MEAN
        steps.append(x)
    return steps


def generate_error_chain(
    key: jax.Array, error_at_step: int
) -> tuple[list[jnp.ndarray], int]:
    """Generate a 5-step chain with one erroneous step at a specified index.

    Steps before the error are "correct" (consistent cluster).
    The error step is sampled from a shifted distribution (inconsistent).
    Steps after the error are also inconsistent (error propagates).

    Args:
        key: JAX PRNG key.
        error_at_step: Index (0-4) of the step where the error is introduced.

    Returns:
        Tuple of (steps, error_at_step) where steps is a list of 5 embeddings.
    """
    steps = []
    for step_idx in range(N_STEPS):
        key, subkey = jrandom.split(key)
        if step_idx < error_at_step:
            # Pre-error: correct/consistent step
            drift = jnp.ones(INPUT_DIM) * (step_idx * 0.05)
            x = jrandom.normal(subkey, (INPUT_DIM,)) * CORRECT_STEP_STD + drift
        elif step_idx == error_at_step:
            # The error step: shifted mean, higher variance
            x = jrandom.normal(subkey, (INPUT_DIM,)) * ERROR_STEP_STD + ERROR_STEP_MEAN
        else:
            # Post-error: inconsistent with pre-error steps (error propagates)
            x = jrandom.normal(subkey, (INPUT_DIM,)) * ERROR_STEP_STD * 0.8 + ERROR_STEP_MEAN * 0.7
        steps.append(x)
    return steps, error_at_step


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_chain_lnn(
    model: LNNConstraintModel,
    steps: list[jnp.ndarray],
) -> list[float]:
    """Evaluate a reasoning chain with the LNN model (with adaptation).

    After processing each step, the LNN's hidden state is updated via adapt().
    This means step k's energy is computed with context from steps 0..k-1.

    Args:
        model: LNNConstraintModel, will be reset before evaluation.
        steps: List of 5 step embeddings.

    Returns:
        List of 5 energy values (one per step).
    """
    model.reset()
    energies = []
    for step_embedding in steps:
        # Compute energy at this step (uses current hidden state context)
        e = float(model.energy(step_embedding))
        energies.append(e)
        # Adapt: incorporate this step's content into the hidden state
        # Future steps will be evaluated with this context
        model.adapt(step_embedding)
    return energies


def evaluate_chain_ising(
    model: IsingModel,
    steps: list[jnp.ndarray],
) -> list[float]:
    """Evaluate a reasoning chain with the static Ising model.

    The Ising model does NOT adapt between steps — each step's energy is
    computed from the same static energy landscape.

    Args:
        model: IsingModel, stateless so no reset needed.
        steps: List of 5 step embeddings.

    Returns:
        List of 5 energy values (one per step).
    """
    return [float(model.energy(step_embedding)) for step_embedding in steps]


def detect_error_step(energies: list[float]) -> int:
    """Return the index of the step with the highest energy.

    High energy = model thinks this step is most anomalous/erroneous.
    """
    return int(jnp.argmax(jnp.array(energies)))


def compute_energy_gap(
    energies: list[float], error_at_step: int
) -> float:
    """Compute energy gap: energy at error step minus mean energy at correct steps.

    A positive gap means the model correctly identified the error step as
    having higher energy than the surrounding correct steps.

    Args:
        energies: List of energy values for the chain.
        error_at_step: Ground-truth index of the erroneous step.

    Returns:
        Energy gap (scalar). Positive = correct detection direction.
    """
    error_energy = energies[error_at_step]
    correct_energies = [e for i, e in enumerate(energies) if i != error_at_step]
    mean_correct = sum(correct_energies) / len(correct_energies) if correct_energies else 0.0
    return error_energy - mean_correct


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 116: LNN adaptive vs Ising static error detection."""
    logger.info("=== Experiment 116: LNN Adaptive Constraint Model ===")

    master_key = jrandom.PRNGKey(SEED)

    # --- Build models ---
    logger.info(f"Building LNN model (input_dim={INPUT_DIM}, hidden_dim={HIDDEN_DIM})")
    lnn_config = LNNConstraintConfig(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        tau_base=0.8,   # Slightly fast adaptation for sensitivity
        dt=0.3,         # Moderate step size
    )
    lnn_model = LNNConstraintModel(lnn_config, key=jrandom.PRNGKey(SEED + 1))

    logger.info(f"Building Ising model (input_dim={INPUT_DIM})")
    ising_model = IsingModel(IsingConfig(input_dim=INPUT_DIM), key=jrandom.PRNGKey(SEED + 2))

    # --- Generate synthetic reasoning chains ---
    logger.info(f"Generating {N_CHAINS} reasoning chains ({N_CHAINS//2} correct, {N_CHAINS//2} with errors)")

    chains_correct: list[dict[str, Any]] = []
    chains_error: list[dict[str, Any]] = []

    # 10 correct chains
    for i in range(N_CHAINS // 2):
        master_key, subkey = jrandom.split(master_key)
        steps = generate_correct_chain(subkey)
        chains_correct.append({
            "chain_id": f"correct_{i:02d}",
            "is_correct": True,
            "error_at_step": None,
            "steps": [s.tolist() for s in steps],
        })

    # 10 error chains — errors distributed across steps 1-3 (steps 2-4 in 1-indexed)
    error_steps = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]  # indices 0-4; errors at 1,2,3
    for i in range(N_CHAINS // 2):
        master_key, subkey = jrandom.split(master_key)
        error_step = error_steps[i % len(error_steps)]
        steps, err_idx = generate_error_chain(subkey, error_at_step=error_step)
        chains_error.append({
            "chain_id": f"error_{i:02d}",
            "is_correct": False,
            "error_at_step": err_idx,
            "steps": [s.tolist() for s in steps],
        })

    all_chains = chains_correct + chains_error
    logger.info(f"Generated {len(all_chains)} chains")

    # --- Evaluate all chains ---
    results_per_chain: list[dict[str, Any]] = []

    lnn_correct_detections = 0
    ising_correct_detections = 0
    lnn_gaps = []
    ising_gaps = []

    for chain_info in all_chains:
        chain_id = chain_info["chain_id"]
        steps = [jnp.array(s) for s in chain_info["steps"]]
        error_at_step = chain_info["error_at_step"]

        # LNN evaluation (adaptive)
        lnn_energies = evaluate_chain_lnn(lnn_model, steps)
        # Ising evaluation (static)
        ising_energies = evaluate_chain_ising(ising_model, steps)

        chain_result: dict[str, Any] = {
            "chain_id": chain_id,
            "is_correct": chain_info["is_correct"],
            "error_at_step": error_at_step,
            "lnn_energies": lnn_energies,
            "ising_energies": ising_energies,
        }

        if not chain_info["is_correct"] and error_at_step is not None:
            # Measure error detection
            lnn_detected_step = detect_error_step(lnn_energies)
            ising_detected_step = detect_error_step(ising_energies)

            lnn_gap = compute_energy_gap(lnn_energies, error_at_step)
            ising_gap = compute_energy_gap(ising_energies, error_at_step)

            lnn_detected_correctly = (lnn_detected_step == error_at_step)
            ising_detected_correctly = (ising_detected_step == error_at_step)

            if lnn_detected_correctly:
                lnn_correct_detections += 1
            if ising_detected_correctly:
                ising_correct_detections += 1

            lnn_gaps.append(lnn_gap)
            ising_gaps.append(ising_gap)

            chain_result.update({
                "lnn_detected_step": lnn_detected_step,
                "ising_detected_step": ising_detected_step,
                "lnn_detected_correctly": lnn_detected_correctly,
                "ising_detected_correctly": ising_detected_correctly,
                "lnn_energy_gap": lnn_gap,
                "ising_energy_gap": ising_gap,
            })

            logger.info(
                f"  {chain_id}: error@step={error_at_step} | "
                f"LNN: detected@{lnn_detected_step} (gap={lnn_gap:+.3f}) | "
                f"Ising: detected@{ising_detected_step} (gap={ising_gap:+.3f})"
            )

        results_per_chain.append(chain_result)

    n_error_chains = N_CHAINS // 2
    lnn_detection_rate = lnn_correct_detections / n_error_chains
    ising_detection_rate = ising_correct_detections / n_error_chains
    lnn_mean_gap = sum(lnn_gaps) / len(lnn_gaps) if lnn_gaps else 0.0
    ising_mean_gap = sum(ising_gaps) / len(ising_gaps) if ising_gaps else 0.0

    logger.info("")
    logger.info("=== Summary ===")
    logger.info(f"LNN detection rate:  {lnn_detection_rate:.1%} ({lnn_correct_detections}/{n_error_chains})")
    logger.info(f"Ising detection rate: {ising_detection_rate:.1%} ({ising_correct_detections}/{n_error_chains})")
    logger.info(f"LNN mean energy gap:  {lnn_mean_gap:+.4f}")
    logger.info(f"Ising mean energy gap: {ising_mean_gap:+.4f}")

    # --- Compute energy evolution statistics ---
    # Average energy per step across correct chains (LNN and Ising)
    lnn_avg_correct_by_step = [0.0] * N_STEPS
    ising_avg_correct_by_step = [0.0] * N_STEPS
    n_correct = sum(1 for r in results_per_chain if r["is_correct"])

    for result in results_per_chain:
        if result["is_correct"]:
            for step_idx in range(N_STEPS):
                lnn_avg_correct_by_step[step_idx] += result["lnn_energies"][step_idx] / max(n_correct, 1)
                ising_avg_correct_by_step[step_idx] += result["ising_energies"][step_idx] / max(n_correct, 1)

    # Average energy per step across error chains (LNN and Ising)
    lnn_avg_error_by_step = [0.0] * N_STEPS
    ising_avg_error_by_step = [0.0] * N_STEPS
    n_error = sum(1 for r in results_per_chain if not r["is_correct"])

    for result in results_per_chain:
        if not result["is_correct"]:
            for step_idx in range(N_STEPS):
                lnn_avg_error_by_step[step_idx] += result["lnn_energies"][step_idx] / max(n_error, 1)
                ising_avg_error_by_step[step_idx] += result["ising_energies"][step_idx] / max(n_error, 1)

    # --- Assemble final results ---
    experiment_results: dict[str, Any] = {
        "experiment": "116_lnn_adaptive",
        "description": "LNN adaptive constraint model vs static Ising for agentic reasoning error detection",
        "config": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_chains": N_CHAINS,
            "n_steps": N_STEPS,
            "seed": SEED,
            "lnn_tau_base": lnn_config.tau_base,
            "lnn_dt": lnn_config.dt,
        },
        "summary": {
            "lnn_detection_rate": lnn_detection_rate,
            "ising_detection_rate": ising_detection_rate,
            "lnn_correct_detections": lnn_correct_detections,
            "ising_correct_detections": ising_correct_detections,
            "n_error_chains": n_error_chains,
            "lnn_mean_energy_gap": lnn_mean_gap,
            "ising_mean_energy_gap": ising_mean_gap,
            "lnn_advantage": lnn_detection_rate - ising_detection_rate,
        },
        "energy_evolution": {
            "lnn_avg_correct_chain_by_step": lnn_avg_correct_by_step,
            "ising_avg_correct_chain_by_step": ising_avg_correct_by_step,
            "lnn_avg_error_chain_by_step": lnn_avg_error_by_step,
            "ising_avg_error_chain_by_step": ising_avg_error_by_step,
        },
        "per_chain_results": results_per_chain,
    }

    # --- Save results ---
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "experiment_116_results.json"

    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("")
    logger.info("=== Key Finding ===")
    if lnn_detection_rate > ising_detection_rate:
        logger.info(
            f"LNN outperforms Ising by {(lnn_detection_rate - ising_detection_rate):.1%} "
            f"on error detection. Adaptation helps!"
        )
    elif lnn_detection_rate == ising_detection_rate:
        logger.info("LNN and Ising perform equally on error detection.")
    else:
        logger.info(
            f"Ising outperforms LNN by {(ising_detection_rate - lnn_detection_rate):.1%}. "
            f"LNN may need more training to show benefit."
        )

    logger.info(
        f"LNN energy gap (error - correct): {lnn_mean_gap:+.4f} | "
        f"Ising energy gap: {ising_mean_gap:+.4f}"
    )


if __name__ == "__main__":
    import jax  # noqa: F401 — ensure JAX is imported before numpy to set platform
    main()
