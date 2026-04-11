#!/usr/bin/env python3
"""Experiment 152: ContinualGibbs vs LNN vs Static Ising on 5-Step Reasoning Chains.

**Researcher summary:**
    Benchmarks orthogonal-gradient continual learning (ContinualGibbsModel) against
    the LNN adaptive model (Exp 116, 10% step-5 accuracy) and static Ising (100%)
    on 5-step math reasoning chains. Target: ContinualGibbs > 80% step-5 accuracy.

**Motivation (from Exp 116 failure analysis):**
    LNN adaptive couplings: 10% step-5 accuracy because adapting within a chain
    destroys previously verified constraints. When step 3 is processed, the LNN
    overwrites the constraint directions learned at steps 1 and 2.

    Exp 139 arxiv scan proposed: use orthogonal parameter updates (LoRA continual
    learning insight). When learning step N, project the gradient onto the null
    space of all prior step gradients. This preserves all previous verifications
    while adapting to new observations.

    ContinualGibbsModel implements this via Gram-Schmidt projection of hidden
    representations (= parameter gradients w.r.t. output_weight).

**Experimental design:**
    - 20 synthetic 5-step reasoning chains (same as Exp 116: same seed, same data)
      - 10 correct chains: embeddings sampled from a consistent Gaussian cluster
      - 10 error chains: errors at steps 1-3 that propagate to step 5
    - Each step is a 16-dimensional synthetic embedding (same as Exp 116)
    - For ContinualGibbs: process steps 0-3 via update_step(), evaluate step 4
    - For LNN: use Exp 116 results (steps 0-3 via adapt(), energy at step 4)
    - For Ising: use Exp 116 results (static, energy at step 4)
    - Metric: binary classification accuracy at step 4 (0-indexed)
      - Binary threshold: median of all step-4 energies
      - Accuracy = max(correct/20, incorrect/20) [sign-agnostic]

**Secondary metrics:**
    - Per-step accuracy degradation (how does accuracy evolve from step 1 to step 5)
    - Energy gap: E(error) - E(correct) at each step
    - Mean energy per step for correct vs error chains

**Output:** results/experiment_152_results.json

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_152_continual.py

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

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX may crash on thrml).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow importing carnot from the python/ directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.continual_gibbs import ContinualGibbsConfig, ContinualGibbsModel
from carnot.models.gibbs import GibbsConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — IDENTICAL to Exp 116 for fair comparison
# ---------------------------------------------------------------------------

INPUT_DIM = 16         # Embedding dimensionality for each reasoning step
HIDDEN_DIM = 8         # Hidden state size (same as LNN's hidden_dim)
N_CHAINS = 20          # Total number of reasoning chains (10 correct + 10 with errors)
N_STEPS = 5            # Steps per reasoning chain
SEED = 2024            # Master RNG seed (SAME as Exp 116 for identical chains)

# Cluster statistics — SAME as Exp 116
CORRECT_STEP_MEAN = 0.0
CORRECT_STEP_STD = 0.3
ERROR_STEP_MEAN = 2.5
ERROR_STEP_STD = 0.5


# ---------------------------------------------------------------------------
# Synthetic chain generation — IDENTICAL to Exp 116 for reproducibility
# ---------------------------------------------------------------------------

def generate_correct_chain(key: jax.Array) -> list[jnp.ndarray]:
    """Generate a 5-step reasoning chain where all steps are consistent.

    Reproduces the same chain generation as Exp 116 for direct comparability.
    Each step is a 16-dim embedding sampled near the origin with slight drift.

    Args:
        key: JAX PRNG key.

    Returns:
        List of 5 step embeddings, each shape (INPUT_DIM,).
    """
    steps = []
    for step_idx in range(N_STEPS):
        key, subkey = jrandom.split(key)
        drift = jnp.ones(INPUT_DIM) * (step_idx * 0.05)
        x = jrandom.normal(subkey, (INPUT_DIM,)) * CORRECT_STEP_STD + drift + CORRECT_STEP_MEAN
        steps.append(x)
    return steps


def generate_error_chain(
    key: jax.Array, error_at_step: int
) -> tuple[list[jnp.ndarray], int]:
    """Generate a 5-step chain with one erroneous step at `error_at_step`.

    Error propagates: steps at and after the error use a shifted distribution.
    Same logic as Exp 116 for direct comparability.

    Args:
        key: JAX PRNG key.
        error_at_step: Index (0-4) of the step where the error is introduced.

    Returns:
        Tuple of (steps, error_at_step).
    """
    steps = []
    for step_idx in range(N_STEPS):
        key, subkey = jrandom.split(key)
        if step_idx < error_at_step:
            drift = jnp.ones(INPUT_DIM) * (step_idx * 0.05)
            x = jrandom.normal(subkey, (INPUT_DIM,)) * CORRECT_STEP_STD + drift
        elif step_idx == error_at_step:
            x = jrandom.normal(subkey, (INPUT_DIM,)) * ERROR_STEP_STD + ERROR_STEP_MEAN
        else:
            # Error propagates: post-error steps are also shifted
            x = jrandom.normal(subkey, (INPUT_DIM,)) * ERROR_STEP_STD * 0.8 + ERROR_STEP_MEAN * 0.7
        steps.append(x)
    return steps, error_at_step


# ---------------------------------------------------------------------------
# ContinualGibbs evaluation
# ---------------------------------------------------------------------------

def evaluate_chain_continual(
    model: ContinualGibbsModel,
    steps: list[jnp.ndarray],
) -> list[float]:
    """Evaluate a reasoning chain with ContinualGibbs (orthogonal updates).

    For each step k:
    - Use steps 0..k-1 as context (via update_step)
    - Evaluate energy at step k
    - Then call update_step for step k to accumulate its constraint

    This gives us a per-step energy trajectory where each step's energy
    is computed AFTER accumulating all prior steps' constraints.

    Args:
        model: ContinualGibbsModel, will be reset before evaluation.
        steps: List of 5 step embeddings.

    Returns:
        List of 5 energy values (one per step), with step k evaluated after
        processing steps 0..k-1 as context.
    """
    model.reset()
    energies = []
    for step_idx, step_embedding in enumerate(steps):
        # Evaluate energy BEFORE updating with this step's embedding.
        # At step 0: no prior context (output_weight = 0, so energy = 0 for all inputs)
        # At step 1: context from step 0
        # At step k: context from steps 0..k-1
        e = float(model.energy(step_embedding))
        energies.append(e)
        # Accumulate this step's constraint for future evaluations.
        model.update_step(step_embedding, step_idx)
    return energies


def compute_step5_energies_continual(
    model: ContinualGibbsModel,
    chains: list[dict[str, Any]],
) -> list[float]:
    """Compute step-4 energies for all chains using ContinualGibbs context accumulation.

    For each chain: process steps 0-3 via update_step(), then evaluate energy at step 4.
    This is the key comparison metric for Exp 152.

    Args:
        model: ContinualGibbsModel.
        chains: List of chain info dicts with 'steps' key.

    Returns:
        List of step-4 energy values, one per chain.
    """
    step5_energies = []
    for chain_info in chains:
        steps = [jnp.array(s) for s in chain_info["steps"]]
        model.reset()

        # Accumulate constraints from steps 0-3 (indices 0..3)
        for step_idx in range(N_STEPS - 1):
            model.update_step(steps[step_idx], step_idx)

        # Evaluate energy at step 4 (index 4 = the 5th step)
        e = float(model.energy(steps[N_STEPS - 1]))
        step5_energies.append(e)

    return step5_energies


# ---------------------------------------------------------------------------
# Binary classification accuracy (sign-agnostic)
# ---------------------------------------------------------------------------

def binary_accuracy_at_threshold(
    energies: list[float],
    labels: list[bool],  # True = error chain, False = correct chain
    threshold: float,
    predict_error_if_high: bool = True,
) -> float:
    """Compute binary classification accuracy using the given energy threshold.

    Predicts "error" for energies above threshold (if predict_error_if_high=True),
    or below threshold (if predict_error_if_high=False).

    Args:
        energies: Energy values per chain.
        labels: True = error chain, False = correct chain.
        threshold: Classification boundary.
        predict_error_if_high: Direction of classification (True = high E → error).

    Returns:
        Fraction of correctly classified chains (accuracy in [0, 1]).
    """
    correct = 0
    for e, label in zip(energies, labels):
        if predict_error_if_high:
            predicted_error = e > threshold
        else:
            predicted_error = e <= threshold
        if predicted_error == label:
            correct += 1
    return correct / len(energies)


def compute_best_accuracy(
    energies: list[float],
    labels: list[bool],
) -> tuple[float, float]:
    """Compute the sign-agnostic binary classification accuracy at the median threshold.

    Uses the median energy as threshold and tries both directions (high-energy =
    error AND low-energy = error), returning the better accuracy and the threshold.

    Args:
        energies: Energy values per chain.
        labels: True = error chain, False = correct chain.

    Returns:
        Tuple of (best_accuracy, threshold_used).
    """
    threshold = float(jnp.median(jnp.array(energies)))
    acc_high = binary_accuracy_at_threshold(energies, labels, threshold, predict_error_if_high=True)
    acc_low = binary_accuracy_at_threshold(energies, labels, threshold, predict_error_if_high=False)
    return max(acc_high, acc_low), threshold


# ---------------------------------------------------------------------------
# Per-step accuracy degradation analysis
# ---------------------------------------------------------------------------

def compute_per_step_accuracy(
    model: ContinualGibbsModel,
    chains: list[dict[str, Any]],
    is_error: list[bool],
) -> list[float]:
    """Compute step-k accuracy for k in {1, 2, 3, 4} using prior steps as context.

    For step k, process steps 0..k-1 via update_step(), evaluate at step k,
    compute binary classification accuracy across all 20 chains.

    Args:
        model: ContinualGibbsModel.
        chains: All 20 chains.
        is_error: Boolean labels (True = error chain).

    Returns:
        List of 4 accuracy values, one per evaluation step (steps 1-4).
    """
    per_step_accuracies = []

    for eval_step in range(1, N_STEPS):  # steps 1, 2, 3, 4 (0-indexed)
        step_energies = []
        for chain_info in chains:
            steps = [jnp.array(s) for s in chain_info["steps"]]
            model.reset()
            # Accumulate context from steps 0..eval_step-1
            for context_step in range(eval_step):
                model.update_step(steps[context_step], context_step)
            # Evaluate at eval_step
            e = float(model.energy(steps[eval_step]))
            step_energies.append(e)

        accuracy, _ = compute_best_accuracy(step_energies, is_error)
        per_step_accuracies.append(accuracy)
        logger.info(f"  ContinualGibbs step-{eval_step + 1} accuracy: {accuracy:.1%}")

    return per_step_accuracies


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 152: ContinualGibbs vs LNN vs Static Ising step-5 accuracy."""
    logger.info("=== Experiment 152: ContinualGibbs Orthogonal Gradient Learning ===")
    logger.info(f"Input dim: {INPUT_DIM}, Hidden dim: {HIDDEN_DIM}")
    logger.info(f"N chains: {N_CHAINS}, N steps: {N_STEPS}, Seed: {SEED}")

    # --- Build ContinualGibbs model ---
    # Same hidden_dims as LNN's hidden_dim for fair comparison
    continual_config = ContinualGibbsConfig(
        gibbs=GibbsConfig(
            input_dim=INPUT_DIM,
            hidden_dims=[HIDDEN_DIM],
            activation="silu",
        ),
        learning_rate=0.1,
    )
    continual_model = ContinualGibbsModel(continual_config, key=jrandom.PRNGKey(SEED + 3))
    logger.info("Built ContinualGibbsModel (input_dim=16, hidden_dims=[8], lr=0.1)")

    # --- Regenerate chains using SAME seed as Exp 116 ---
    # This ensures we're comparing on identical data.
    logger.info("Generating chains using Exp 116 seed (SEED=2024) for direct comparison...")
    master_key = jrandom.PRNGKey(SEED)

    all_chains: list[dict[str, Any]] = []

    # 10 correct chains
    for i in range(N_CHAINS // 2):
        master_key, subkey = jrandom.split(master_key)
        steps = generate_correct_chain(subkey)
        all_chains.append({
            "chain_id": f"correct_{i:02d}",
            "is_correct": True,
            "error_at_step": None,
            "steps": [s.tolist() for s in steps],
        })

    # 10 error chains — same distribution as Exp 116
    error_steps_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]
    for i in range(N_CHAINS // 2):
        master_key, subkey = jrandom.split(master_key)
        error_step = error_steps_seq[i % len(error_steps_seq)]
        steps, err_idx = generate_error_chain(subkey, error_at_step=error_step)
        all_chains.append({
            "chain_id": f"error_{i:02d}",
            "is_correct": False,
            "error_at_step": err_idx,
            "steps": [s.tolist() for s in steps],
        })

    is_error = [not c["is_correct"] for c in all_chains]
    logger.info(f"Generated {N_CHAINS} chains ({sum(not e for e in is_error)} correct, {sum(is_error)} error)")

    # --- Evaluate ContinualGibbs per-step trajectory ---
    logger.info("\n--- ContinualGibbs: full 5-step energy trajectory ---")
    continual_all_energies: list[list[float]] = []
    for chain_info in all_chains:
        steps = [jnp.array(s) for s in chain_info["steps"]]
        energies = evaluate_chain_continual(continual_model, steps)
        continual_all_energies.append(energies)

    # --- Evaluate ContinualGibbs step-5 accuracy (primary metric) ---
    logger.info("\n--- ContinualGibbs: step-5 accuracy (steps 0-3 as context) ---")
    continual_step5_energies = compute_step5_energies_continual(continual_model, all_chains)
    continual_step5_accuracy, continual_threshold = compute_best_accuracy(
        continual_step5_energies, is_error
    )
    logger.info(
        f"ContinualGibbs step-5 accuracy: {continual_step5_accuracy:.1%} "
        f"(threshold={continual_threshold:.4f})"
    )

    # --- ContinualGibbs per-step accuracy degradation ---
    logger.info("\n--- ContinualGibbs: per-step accuracy (steps 1-5) ---")
    continual_per_step_acc = compute_per_step_accuracy(continual_model, all_chains, is_error)

    # --- Load Exp 116 results for LNN and Ising comparison ---
    logger.info("\n--- Loading Exp 116 results for LNN and Ising step-5 accuracy ---")
    exp116_path = _ROOT / "results" / "experiment_116_results.json"
    if exp116_path.exists():
        with open(exp116_path) as f:
            exp116_results = json.load(f)

        # Extract step-4 energies from Exp 116 per-chain results
        lnn_step5_energies = [
            r["lnn_energies"][4] for r in exp116_results["per_chain_results"]
        ]
        ising_step5_energies = [
            r["ising_energies"][4] for r in exp116_results["per_chain_results"]
        ]

        # Labels from Exp 116 (should match since same chain generation order)
        exp116_is_error = [
            not r["is_correct"] for r in exp116_results["per_chain_results"]
        ]

        lnn_step5_accuracy, lnn_threshold = compute_best_accuracy(lnn_step5_energies, exp116_is_error)
        ising_step5_accuracy, ising_threshold = compute_best_accuracy(
            ising_step5_energies, exp116_is_error
        )

        logger.info(f"LNN step-5 accuracy (from Exp 116):   {lnn_step5_accuracy:.1%}")
        logger.info(f"Ising step-5 accuracy (from Exp 116): {ising_step5_accuracy:.1%}")

        lnn_available = True
    else:
        logger.warning(f"Exp 116 results not found at {exp116_path}. Using N/A for LNN/Ising.")
        lnn_step5_energies = []
        ising_step5_energies = []
        lnn_step5_accuracy = float("nan")
        ising_step5_accuracy = float("nan")
        lnn_threshold = float("nan")
        ising_threshold = float("nan")
        lnn_available = False

    # --- Summary ---
    logger.info("\n=== Experiment 152 Summary ===")
    logger.info(
        f"{'Model':<20} {'Step-5 Accuracy':>16} {'vs LNN':>10} {'Target Met':>12}"
    )
    logger.info("-" * 62)
    lnn_acc_str = f"{lnn_step5_accuracy:.1%}" if lnn_available else "N/A"
    ising_acc_str = f"{ising_step5_accuracy:.1%}" if lnn_available else "N/A"
    continual_vs_lnn = (
        f"+{(continual_step5_accuracy - lnn_step5_accuracy):.1%}"
        if lnn_available else "N/A"
    )
    target_met = "YES" if continual_step5_accuracy >= 0.80 else "NO"

    logger.info(f"{'Static Ising (Exp 116)':<20} {ising_acc_str:>16} {'---':>10} {'---':>12}")
    logger.info(f"{'LNN (Exp 116)':<20} {lnn_acc_str:>16} {'---':>10} {'---':>12}")
    logger.info(
        f"{'ContinualGibbs':<20} {continual_step5_accuracy:.1%}{' ':>9} {continual_vs_lnn:>10} {target_met:>12}"
    )
    logger.info("")
    logger.info(f"Target: ContinualGibbs > 80% — {target_met}")
    logger.info(
        f"ContinualGibbs per-step accuracy: "
        + " | ".join(f"step{i+2}={a:.0%}" for i, a in enumerate(continual_per_step_acc))
    )

    # --- Compute energy gap statistics ---
    continual_correct_step5 = [
        e for e, err in zip(continual_step5_energies, is_error) if not err
    ]
    continual_error_step5 = [
        e for e, err in zip(continual_step5_energies, is_error) if err
    ]
    continual_mean_correct_step5 = float(jnp.mean(jnp.array(continual_correct_step5))) if continual_correct_step5 else float("nan")
    continual_mean_error_step5 = float(jnp.mean(jnp.array(continual_error_step5))) if continual_error_step5 else float("nan")
    continual_energy_gap = continual_mean_error_step5 - continual_mean_correct_step5

    logger.info(
        f"ContinualGibbs energy gap (error - correct) at step 5: {continual_energy_gap:+.4f}"
    )
    logger.info(
        f"  mean E(correct step 5) = {continual_mean_correct_step5:.4f}"
    )
    logger.info(
        f"  mean E(error step 5)   = {continual_mean_error_step5:.4f}"
    )

    # --- Per-chain results ---
    per_chain_results = []
    for i, chain_info in enumerate(all_chains):
        chain_result: dict[str, Any] = {
            "chain_id": chain_info["chain_id"],
            "is_correct": chain_info["is_correct"],
            "error_at_step": chain_info["error_at_step"],
            "continual_energies": continual_all_energies[i],
            "continual_step5_energy": continual_step5_energies[i],
        }
        if lnn_available:
            chain_result["lnn_step5_energy"] = lnn_step5_energies[i]
            chain_result["ising_step5_energy"] = ising_step5_energies[i]
        per_chain_results.append(chain_result)

    # --- Assemble final results ---
    experiment_results: dict[str, Any] = {
        "experiment": "152_continual_gibbs",
        "description": "ContinualGibbs orthogonal gradient vs LNN (Exp 116) and Ising on 5-step chains",
        "config": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_chains": N_CHAINS,
            "n_steps": N_STEPS,
            "seed": SEED,
            "learning_rate": continual_config.learning_rate,
            "activation": continual_config.gibbs.activation,
        },
        "summary": {
            "continual_step5_accuracy": continual_step5_accuracy,
            "lnn_step5_accuracy": lnn_step5_accuracy if lnn_available else None,
            "ising_step5_accuracy": ising_step5_accuracy if lnn_available else None,
            "target_80pct_met": continual_step5_accuracy >= 0.80,
            "continual_vs_lnn_improvement": (
                continual_step5_accuracy - lnn_step5_accuracy if lnn_available else None
            ),
            "continual_energy_gap_step5": continual_energy_gap,
            "continual_mean_correct_step5_energy": continual_mean_correct_step5,
            "continual_mean_error_step5_energy": continual_mean_error_step5,
            "continual_classification_threshold": continual_threshold,
            "continual_per_step_accuracies": continual_per_step_acc,
        },
        "lnn_exp116_available": lnn_available,
        "per_chain_results": per_chain_results,
    }

    # --- Save results ---
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "experiment_152_results.json"

    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # --- Final verdict ---
    logger.info("\n=== Key Finding ===")
    if continual_step5_accuracy >= 0.80:
        logger.info(
            f"ContinualGibbs achieves {continual_step5_accuracy:.1%} step-5 accuracy — "
            f"TARGET MET (>80%). Orthogonal gradient projection successfully preserves "
            f"prior-step constraints."
        )
    else:
        logger.info(
            f"ContinualGibbs achieves {continual_step5_accuracy:.1%} step-5 accuracy — "
            f"target 80% NOT YET MET. Consider tuning learning_rate or hidden_dims."
        )
    if lnn_available:
        logger.info(
            f"Improvement over LNN: {continual_step5_accuracy - lnn_step5_accuracy:+.1%} "
            f"({lnn_step5_accuracy:.1%} → {continual_step5_accuracy:.1%})"
        )


if __name__ == "__main__":
    main()
