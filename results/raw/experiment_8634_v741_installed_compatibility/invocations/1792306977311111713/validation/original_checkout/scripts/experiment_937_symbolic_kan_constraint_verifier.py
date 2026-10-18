#!/usr/bin/env python3
"""Experiment 937 — Symbolic-KAN Arithmetic Constraint Verifier.

**Goal:**
    Compare SymbolicKAN (arXiv 2603.23854) against the standard KAN baseline
    (Exp 910, AUC = 0.2208) on synthetic arithmetic violation detection.
    Target: AUC > 0.70 on a held-out evaluation set.

**What we do:**
    1. Generate 200 synthetic (correct_CoT, hallucinated_CoT) pairs.
       Each pair is encoded as a fixed-length feature vector (16 dims) that
       captures arithmetic properties: operand values, claimed result,
       delta from correct result, sign of delta, etc.
    2. Train SymbolicKAN on 160 pairs (160 correct, 160 hallucinated).
    3. Evaluate on 40 held-out pairs.
    4. Compare AUC vs Exp 910 baseline.

**Honest verdict mapping:**
    auc_symbolic > 0.70  → 'symbolic_kan_viable'
    0.60 < auc_symbolic ≤ 0.70 → 'symbolic_kan_marginal'
    otherwise            → 'symbolic_kan_no_improvement'

CPU-only experiment: no GPU required.

Spec: REQ-MODEL-030, SCENARIO-MODEL-015.
"""

import json
import math
import os
import random
import sys
import time

# Ensure repo root is on sys.path so experiment_template imports work.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from scripts.experiment_template import ExperimentTemplate
from python.carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _make_feature_vector(
    a: float,
    b: float,
    op: str,
    claimed: float,
    correct: float,
) -> np.ndarray:
    """Encode one arithmetic statement as a 16-dim feature vector.

    Features (in order):
      0  a  (first operand, normalised to [-1, 1])
      1  b  (second operand, normalised to [-1, 1])
      2  claimed result (normalised)
      3  correct result (normalised)
      4  delta = claimed - correct (normalised)
      5  sign of delta
      6  |delta| (absolute error)
      7  op == ADD as float (1.0 if add, else 0.0)
      8  op == MUL as float
      9  op == CMP as float (1.0 if comparison, else 0.0)
      10 claimed > 0 (1.0 if claimed positive)
      11 correct > 0 (1.0 if correct positive)
      12 a * b (product term, normalised)
      13 a + b (sum term, normalised)
      14 |a - b| (difference term)
      15 claimed == correct as float (EQ flag)
    """
    # Normalise to [-1, 1] using a scale of 20
    scale = 20.0
    a_n = max(-1.0, min(1.0, a / scale))
    b_n = max(-1.0, min(1.0, b / scale))
    c_n = max(-1.0, min(1.0, claimed / scale))
    k_n = max(-1.0, min(1.0, correct / scale))
    delta = claimed - correct
    delta_n = max(-1.0, min(1.0, delta / scale))
    sign_delta = math.copysign(1.0, delta) if delta != 0 else 0.0
    abs_delta = abs(delta) / scale
    is_add = 1.0 if op == "ADD" else 0.0
    is_mul = 1.0 if op == "MUL" else 0.0
    is_cmp = 1.0 if op == "CMP" else 0.0
    claimed_pos = 1.0 if claimed > 0 else 0.0
    correct_pos = 1.0 if correct > 0 else 0.0
    prod = max(-1.0, min(1.0, a * b / (scale * scale)))
    summ = max(-1.0, min(1.0, (a + b) / scale))
    diff = max(-1.0, min(1.0, abs(a - b) / scale))
    eq_flag = 1.0 if claimed == correct else 0.0

    return np.array(
        [
            a_n,
            b_n,
            c_n,
            k_n,
            delta_n,
            sign_delta,
            abs_delta,
            is_add,
            is_mul,
            is_cmp,
            claimed_pos,
            correct_pos,
            prod,
            summ,
            diff,
            eq_flag,
        ],
        dtype=np.float32,
    )


def generate_synthetic_pairs(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate n (correct_features, hallucinated_features) pairs.

    Arithmetic operations: ADD (a+b), MUL (a*b).
    Hallucinations: off-by-one to off-by-5 errors, or wrong-sign errors.

    Returns:
        xs_correct:  shape (n, 16)
        xs_incorrect: shape (n, 16) — same (a, b, op) but claimed result is wrong
    """
    rng = random.Random(seed)
    xs_correct = []
    xs_incorrect = []

    ops = ["ADD", "MUL"]
    for i in range(n):
        op = rng.choice(ops)
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)

        if op == "ADD":
            correct = float(a + b)
        else:
            correct = float(a * b)

        # Hallucination: add a non-zero error (1 to 5, random sign)
        error = rng.randint(1, 5) * rng.choice([-1, 1])
        claimed_wrong = correct + error

        feat_correct = _make_feature_vector(float(a), float(b), op, correct, correct)
        feat_wrong = _make_feature_vector(float(a), float(b), op, claimed_wrong, correct)

        xs_correct.append(feat_correct)
        xs_incorrect.append(feat_wrong)

    return np.array(xs_correct, dtype=np.float32), np.array(xs_incorrect, dtype=np.float32)


# ---------------------------------------------------------------------------
# AUC computation (ROC)
# ---------------------------------------------------------------------------


def compute_auc(
    model: SymbolicKANModel,
    xs_correct: np.ndarray,
    xs_incorrect: np.ndarray,
) -> float:
    """Compute AUC-ROC for binary classification: correct vs incorrect.

    The model assigns energy to each sample.  We want correct samples to have
    lower energy than incorrect samples.  We treat:
        - correct samples as "negative class" (label 0, expect low energy)
        - incorrect samples as "positive class" (label 1, expect high energy)

    A random classifier gets AUC = 0.5.
    A perfect classifier gets AUC = 1.0.
    AUC > 0.70 is our target for 'viable'.

    Implementation: trapezoidal rule over the ROC curve using energy as the
    decision variable.  All samples are sorted by energy (ascending); we
    sweep the threshold and compute TPR/FPR at each step.
    """
    n = len(xs_correct)
    energies_correct = model.energy_batch(xs_correct)
    energies_incorrect = model.energy_batch(xs_incorrect)

    # Labels: 0 = correct (negative class), 1 = incorrect (positive class)
    scores = np.concatenate([energies_correct, energies_incorrect])
    labels = np.concatenate([np.zeros(n), np.ones(n)])

    # Sort by score descending (high energy first → predict incorrect)
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    P = n  # total positives (incorrect)
    N = n  # total negatives (correct)

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0
    return float(auc)


# ---------------------------------------------------------------------------
# Interpretability examples
# ---------------------------------------------------------------------------


def make_interpretability_examples(
    model: SymbolicKANModel,
    xs_correct: np.ndarray,
    xs_incorrect: np.ndarray,
) -> list[dict]:
    """Generate 3 interpretability examples showing what top nodes check.

    For each of the top 3 most-used symbolic labels, find the node that uses
    that label and show what it computes on a real example.

    SCENARIO-MODEL-015.
    """
    examples = []
    top = model.top_labels()
    used_labels = set()

    for node_idx in range(model.config.n_nodes):
        label = model.symbolic_labels[node_idx]
        if label not in used_labels and len(examples) < 3:
            used_labels.add(label)
            desc = model.describe_node(node_idx)
            i1 = model.in1[node_idx]
            i2 = model.in2[node_idx]

            # Show on first eval sample
            xc = xs_correct[0]
            xi = xs_incorrect[0]

            from python.carnot.models.symbolic_kan import VOCAB

            sym_fn = VOCAB[label]
            examples.append(
                {
                    "node": node_idx,
                    "label": label,
                    "description": desc,
                    "correct_sample": {
                        "feat_i1": float(xc[i1]),
                        "feat_i2": float(xc[i2]),
                        "symbolic_output": float(sym_fn(float(xc[i1]), float(xc[i2]))),
                    },
                    "incorrect_sample": {
                        "feat_i1": float(xi[i1]),
                        "feat_i2": float(xi[i2]),
                        "symbolic_output": float(sym_fn(float(xi[i1]), float(xi[i2]))),
                    },
                }
            )

    return examples


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 937: Symbolic-KAN vs standard KAN AUC comparison."""
    tmpl = ExperimentTemplate(
        exp_id=937,
        title="Symbolic-KAN Arithmetic Constraint Verifier",
        deliverable="results/experiment_937_symbolic_kan_constraint_verifier.json",
    )
    tmpl.setup()

    t_start = time.time()

    with tmpl.phase("data_generation"):
        xs_correct_all, xs_incorrect_all = generate_synthetic_pairs(n=200, seed=42)

        # Train/eval split: 160 train, 40 eval
        xs_correct_train = xs_correct_all[:160]
        xs_incorrect_train = xs_incorrect_all[:160]
        xs_correct_eval = xs_correct_all[160:]
        xs_incorrect_eval = xs_incorrect_all[160:]

    with tmpl.phase("training"):
        config = SymbolicKANConfig(
            input_dim=16,
            n_nodes=8,
            label_update_interval=10,
            residual_amp=0.05,
            lr=0.02,
            n_segments=8,
        )
        model = SymbolicKANModel(config, seed=42)
        loss_history = model.train(
            xs_correct_train,
            xs_incorrect_train,
            n_epochs=60,
        )

    with tmpl.phase("evaluation"):
        auc_symbolic = compute_auc(model, xs_correct_eval, xs_incorrect_eval)

        # Baseline from Exp 910 (standard KAN, synthetic data AUC)
        # Exp 910 reports baseline_auc=0.1584, post_refinement_auc=0.2208
        auc_standard = 0.2208
        delta_auc = auc_symbolic - auc_standard

        top_labels = model.top_labels()
        label_counts = model.label_counts()
        interp_examples = make_interpretability_examples(model, xs_correct_eval, xs_incorrect_eval)

    if auc_symbolic > 0.70:
        honest_verdict = "symbolic_kan_viable"
    elif auc_symbolic > 0.60:
        honest_verdict = "symbolic_kan_marginal"
    else:
        honest_verdict = "symbolic_kan_no_improvement"

    result = tmpl.build_result(
        {
            "auc_standard": auc_standard,
            "auc_symbolic": round(auc_symbolic, 4),
            "delta_auc": round(delta_auc, 4),
            "honest_verdict": honest_verdict,
            "top_symbolic_labels": top_labels,
            "label_counts": label_counts,
            "interpretability_examples": interp_examples,
            "final_train_loss": round(loss_history[-1], 4) if loss_history else None,
            "n_train_pairs": 160,
            "n_eval_pairs": 40,
            "n_epochs": 60,
            "symbolic_kan_config": {
                "input_dim": config.input_dim,
                "n_nodes": config.n_nodes,
                "label_update_interval": config.label_update_interval,
                "residual_amp": config.residual_amp,
                "lr": config.lr,
            },
            "baseline_exp": 910,
            "models_used": ["synthetic_embeddings_cpu_only"],
            "tier": "FR-11 Tier 4",
        },
        status="success",
    )

    deliverable_path = "results/experiment_937_symbolic_kan_constraint_verifier.json"
    os.makedirs("results", exist_ok=True)
    with open(deliverable_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"[Exp 937] auc_symbolic={auc_symbolic:.4f}  auc_standard={auc_standard:.4f}  "
        f"delta={delta_auc:+.4f}  verdict={honest_verdict}"
    )
    print(f"[Exp 937] top_labels={top_labels}")
    print(f"[Exp 937] Written: {deliverable_path}")


if __name__ == "__main__":
    main()
