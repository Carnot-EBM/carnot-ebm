#!/usr/bin/env python3
"""Experiment 534: Potts Machine Verifier — q=3 constraint encoding vs binary IsingEBM.

**Researcher summary:**
    IsingEBM treats constraint verification as binary: either satisfied (+1) or violated (-1).
    This loses partial-credit signal: a multi-step arithmetic proof with one wrong sub-step
    is "violated" same as a completely nonsensical answer.

    The Potts Machine (arXiv 2602.04200) generalizes Ising to q discrete states per spin,
    enabling 3-class encoding: correct (0) / partial (1) / violated (2).  The key property
    is that sparse coupling structure is preserved under mean-field optimization, keeping the
    model FPGA-compatible for the KV260 hardware target.

    This experiment validates PottsMachineVerifier(q=3) on synthetic arithmetic constraint
    examples:
    - 100 'correct' examples: all arithmetic correct
    - 100 'partial' examples: one small arithmetic error in a sub-step
    - 100 'violated' examples: clear arithmetic contradiction

    We compare:
    - PottsMachineVerifier(n_spins=16, q=3): 3-class AUROC
    - IsingEBM baseline (binary: correct vs not-correct): binary AUROC

    The 'partial_class_accuracy' metric shows how well Potts identifies partial-credit
    examples that IsingEBM cannot distinguish from full violations.

**Expected outcome:**
    potts_3class_auroc should be competitive with or exceed ising_binary_auroc, while
    also capturing partial-credit examples that binary Ising cannot classify.

**Outputs:**
    results/experiment_534_potts_machine_verifier.json

Spec: REQ-VERIFY-106, REQ-VERIFY-107, REQ-VERIFY-108,
      SCENARIO-VERIFY-142, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any JAX/CUDA import (RETRO-022 fix)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import random

import jax
import jax.numpy as jnp
import numpy as np

from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.potts_machine import PottsMachineVerifier
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SPINS = 16        # Potts spins per constraint example
N_EXAMPLES = 300    # Total examples: 100 per class
N_TRAIN = 240       # 80/20 split
N_TEST = 60
Q = 3               # States: 0=correct, 1=partial, 2=violated


# ---------------------------------------------------------------------------
# Synthetic constraint example generation
# ---------------------------------------------------------------------------


def _make_arithmetic_example(kind: str, idx: int, dim: int = N_SPINS) -> tuple[jax.Array, int]:
    """Generate a synthetic arithmetic constraint example as a spin vector.

    Why hash-encoding rather than learned embeddings:
        This experiment validates the Potts energy landscape, not a text encoder.
        Using a deterministic hash encoding isolates the energy function's contribution:
        any classification improvement must come from the Potts energy surface.

    The encoding is:
    - Encode the arithmetic problem string into a float vector of shape (dim,)
    - Class 0 (correct): A + B = C where C is correct, encoded normally
    - Class 1 (partial): A + B = C + 1 (off by one — partial credit)
    - Class 2 (violated): A + B = C + 10 (clear contradiction)

    Returns:
        (config, label) where config is shape (dim,) float array and label is 0/1/2.
    """
    rng = random.Random(idx * 31 + hash(kind) % 997)
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    correct_sum = a + b

    if kind == "correct":
        # Correct: A + B = correct_sum
        statement = f"{a}+{b}={correct_sum}"
        label = 0
    elif kind == "partial":
        # Partial: off by one — one small arithmetic error in a sub-step
        wrong_sum = correct_sum + 1
        statement = f"{a}+{b}={wrong_sum}"
        label = 1
    else:  # "violated"
        # Violated: clear arithmetic contradiction (off by 10+)
        wrong_sum = correct_sum + rng.randint(10, 20)
        statement = f"{a}+{b}={wrong_sum}"
        label = 2

    # Deterministic hash encoding: map statement chars to spin float vector
    arr = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(statement):
        idx_pos = (ord(ch) + i * 7) % dim
        arr[idx_pos] += 1.0
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm

    # Convert float encoding to integer spin state for Potts: discretize to {0,1,2}
    # Bin the continuous encoding into q=3 states for Potts
    spin_config = np.zeros(dim, dtype=np.int32)
    for i in range(dim):
        # Bin float value into {0, 1, 2} based on terciles [0, 0.33, 0.67, 1.0]
        # (after renormalizing to [0, 1])
        v = float(arr[i])
        if v < 0.33:
            spin_config[i] = 0
        elif v < 0.67:
            spin_config[i] = 1
        else:
            spin_config[i] = 2

    return jnp.array(spin_config, dtype=jnp.int32), label


def _make_ising_float_config(example: jax.Array) -> jax.Array:
    """Convert Potts integer config to Ising float config for IsingEBM comparison.

    The Ising model uses float inputs: map states {0,1,2} to float values
    {-1.0, 0.0, +1.0} so that correct (0) maps to -1, partial (1) to 0,
    violated (2) to +1.  This makes the binary split between correct (low energy)
    and not-correct (high energy) correspond to negative vs positive spin.
    """
    return (example.astype(jnp.float32) - 1.0)  # {0,1,2} -> {-1,0,+1}


# ---------------------------------------------------------------------------
# AUROC computation (no scikit-learn dependency)
# ---------------------------------------------------------------------------


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUROC via the trapezoidal rule.

    Lower score = model predicts "positive class" (correct for IsingEBM).
    Labels: 1 = positive (correct), 0 = negative (not correct).

    Why reimplement rather than import sklearn:
        sklearn is not in Carnot's required dependencies, and this experiment
        runs on CPU-only machines where scipy may also be absent.  The
        trapezoidal AUROC is 6 lines and is correct for finite sorted thresholds.
    """
    # Sort by score ascending (lower score = more likely positive)
    order = np.argsort(scores)
    sorted_labels = labels[order]
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case

    # TPR and FPR at each threshold
    tps = np.cumsum(sorted_labels[::-1])[::-1]  # cumsum from right to left
    fps = np.cumsum(1 - sorted_labels[::-1])[::-1]
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Append (0,0) and (1,1) endpoints
    tpr = np.concatenate([[1.0], tpr, [0.0]])
    fpr = np.concatenate([[1.0], fpr, [0.0]])

    # Trapezoidal integration
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auroc = float(trapz(tpr, fpr))
    return abs(auroc)  # abs in case of inverted ordering


def _multiclass_auroc_ovr(scores_matrix: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Compute macro-averaged one-vs-rest AUROC for q-class predictions.

    For each class c: compute AUROC with c as positive vs all others as negative.
    Return the macro average across all q classes.

    This is the standard multi-class AUROC extension (Hand & Till 2001).
    Lower energy for class c = model predicts class c.
    """
    aurocs = []
    for c in range(n_classes):
        # Scores for class c: use -energy_c as the score (lower energy = higher score)
        scores_c = -scores_matrix[:, c]
        bin_labels = (labels == c).astype(np.int32)
        aurocs.append(_binary_auroc(scores_c, bin_labels))
    return float(np.mean(aurocs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 534: Potts Machine Verifier vs IsingEBM baseline."""
    tmpl = ExperimentTemplate(
        534,
        "Potts Machine Verifier",
        "results/experiment_534_potts_machine_verifier.json",
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(534, timeout_minutes=25)
    output_path = _REPO_ROOT / "results" / "experiment_534_potts_machine_verifier.json"
    guard = DeliverableGuard(str(output_path))

    _log.info("Exp 534: generating %d synthetic constraint examples", N_EXAMPLES)

    # --- Generate examples ---
    all_configs_potts = []
    all_configs_ising = []
    all_labels = []

    kinds = ["correct"] * 100 + ["partial"] * 100 + ["violated"] * 100
    for i, kind in enumerate(kinds):
        config_potts, label = _make_arithmetic_example(kind, idx=i)
        config_ising = _make_ising_float_config(config_potts)
        all_configs_potts.append(config_potts)
        all_configs_ising.append(config_ising)
        all_labels.append(label)

    configs_potts = jnp.stack(all_configs_potts)   # (300, 16)
    configs_ising = jnp.stack(all_configs_ising)   # (300, 16)
    labels_np = np.array(all_labels, dtype=np.int32)

    # --- Train/test split (same for both models) ---
    # Use first 240 for training, last 60 for test (stratified by class)
    # Classes are in blocks of 100, so take 80 from each class for train
    train_idx = list(range(0, 80)) + list(range(100, 180)) + list(range(200, 280))
    test_idx = list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))

    train_potts = configs_potts[jnp.array(train_idx)]
    test_potts = configs_potts[jnp.array(test_idx)]
    test_ising = configs_ising[jnp.array(test_idx)]
    test_labels = labels_np[test_idx]

    # Split train by class
    train_correct = train_potts[:80]    # first 80 train are class 0
    train_partial = train_potts[80:160] # next 80 train are class 1
    train_violated = train_potts[160:]  # last 80 train are class 2

    # --- Train PottsMachineVerifier ---
    _log.info("Exp 534: training PottsMachineVerifier(n_spins=%d, q=%d)", N_SPINS, Q)
    potts_model = PottsMachineVerifier(n_spins=N_SPINS, q=Q, key=jax.random.PRNGKey(534))
    potts_model.fit_cd(
        correct_configs=train_correct,
        violated_configs=train_violated,
        partial_configs=train_partial,
        n_steps=50,
        lr=0.01,
    )
    _log.info("Exp 534: PottsMachineVerifier training complete")

    # --- Train IsingEBM baseline (binary: correct=0 vs not-correct=1/2) ---
    _log.info("Exp 534: training IsingEBM baseline (binary correct vs not-correct)")
    ising_model = IsingModel(IsingConfig(input_dim=N_SPINS), key=jax.random.PRNGKey(534))

    # CD training for Ising: correct (label=0) as positive, violated (label=2) as negative
    # We use a simple gradient-sign CD update manually
    ising_lr = 0.005
    ising_train_configs_ising = configs_ising[jnp.array(train_idx)]
    ising_train_labels = labels_np[train_idx]

    J_np = np.array(ising_model.coupling)
    b_np = np.array(ising_model.bias)

    rng_ising = np.random.default_rng(534)
    for step in range(50):
        # Sample one correct and one violated example
        correct_pool = np.where(ising_train_labels == 0)[0]
        violated_pool = np.where(ising_train_labels == 2)[0]
        pi = rng_ising.choice(correct_pool)
        ni = rng_ising.choice(violated_pool)

        pos_x = np.array(ising_train_configs_ising[pi])
        neg_x = np.array(ising_train_configs_ising[ni])

        # Gradient of E = -0.5 * x^T J x - b^T x w.r.t. J: dE/dJ = -0.5 * x x^T (sym)
        # CD update: J += lr * (neg_outer - pos_outer) to lower E(pos), raise E(neg)
        pos_outer = np.outer(pos_x, pos_x)
        neg_outer = np.outer(neg_x, neg_x)
        J_np += ising_lr * (neg_outer - pos_outer)

        # Gradient w.r.t. b: dE/db = -x
        b_np += ising_lr * (neg_x - pos_x)

    ising_model.coupling = jnp.array(J_np)
    ising_model.bias = jnp.array(b_np)
    _log.info("Exp 534: IsingEBM training complete")

    # --- Evaluate PottsMachineVerifier ---
    _log.info("Exp 534: evaluating PottsMachineVerifier on %d test examples", len(test_idx))

    # Compute energies for each class assignment per test example
    potts_class_energies = np.zeros((len(test_idx), Q))
    potts_predictions = []
    for i, idx in enumerate(range(len(test_idx))):
        config = test_potts[idx]
        for cls in range(Q):
            class_config = jnp.full((N_SPINS,), cls, dtype=jnp.int32)
            potts_class_energies[i, cls] = float(potts_model.energy(class_config))
        potts_predictions.append(potts_model.predict_class(config))

    potts_predictions_np = np.array(potts_predictions, dtype=np.int32)
    potts_3class_auroc = _multiclass_auroc_ovr(potts_class_energies, test_labels, Q)

    # Partial class accuracy: how often does Potts correctly identify class 1 (partial)?
    partial_mask = test_labels == 1
    if np.sum(partial_mask) > 0:
        partial_correct = np.sum(potts_predictions_np[partial_mask] == 1)
        partial_class_accuracy = float(partial_correct) / float(np.sum(partial_mask))
    else:
        partial_class_accuracy = 0.0

    # --- Evaluate IsingEBM (binary AUROC: class 0 vs rest) ---
    _log.info("Exp 534: evaluating IsingEBM baseline")
    ising_energies = np.array([float(ising_model.energy(test_ising[i])) for i in range(len(test_idx))])
    ising_bin_labels = (test_labels == 0).astype(np.int32)
    ising_binary_auroc = _binary_auroc(ising_energies, ising_bin_labels)

    _log.info(
        "Exp 534: potts_3class_auroc=%.4f, ising_binary_auroc=%.4f, partial_acc=%.4f",
        potts_3class_auroc, ising_binary_auroc, partial_class_accuracy
    )

    # --- Build artifact ---
    potts_viable = bool(potts_3class_auroc >= ising_binary_auroc)
    honest_verdict = "potts_advantage" if potts_viable else "no_advantage"

    artifact = tmpl.build_result(
        {
            "potts_3class_auroc": round(potts_3class_auroc, 4),
            "ising_binary_auroc": round(ising_binary_auroc, 4),
            "partial_class_accuracy": round(partial_class_accuracy, 4),
            "potts_viable": potts_viable,
            "fpga_path_note": "PottsMachineVerifier sparse coupling compatible with KV260 — q=3 sampler needs Verilog extension of existing FPGA Ising module",
            "honest_verdict": honest_verdict,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_spins": N_SPINS,
            "q": Q,
            "potts_train_steps": 50,
            "ising_train_steps": 50,
        },
        status="success",
    )
    # Set schema name after build_result (which auto-sets schema to sorted key list)
    artifact["schema"] = "carnot.potts_machine.v1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info("Exp 534: artifact written to %s", output_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
