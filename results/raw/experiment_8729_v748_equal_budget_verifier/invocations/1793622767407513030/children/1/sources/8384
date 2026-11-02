#!/usr/bin/env python3
"""Exp 947 — DRIFTProbe v3 Depth-Recurrent Attention Pooling (CPU).

WHY THIS EXPERIMENT:
    Experiments 911 (tier0i_marginal) and 923 (tier0i_no_improvement) both failed
    to beat the uniform-weight ensemble baseline for drift-based hallucination detection.
    The root cause (per arXiv 2604.17121) is that transformer state is non-local:
    signals propagate across layers, so fixed-layer and uniform-weight probes miss
    the actual drift-bearing layers for any given input.

    The fix (arXiv 2604.13386): learned attention pooling over all layers, so the
    model discovers *which* layers carry drift signal for this distribution.

    This experiment validates the fix on synthetic data:
    - 100 "correct" samples: hidden states with small drift across all layers.
    - 100 "incorrect" samples: hidden states with large drift injected at alternating layers.
    - Target: probe_auc > 0.50 (above random), ideally > Exp 923 baseline of 0.5625.

SPEC: REQ-PROBE-010, SCENARIO-PROBE-015
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.drift_probe_v3 import DRIFTProbeV3

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 947
TITLE = "DRIFTProbe v3 — Depth-Recurrent Attention Pooling (CPU, Synthetic)"
DELIVERABLE = "results/experiment_947_driftprobe_v3_depth_recurrent.json"

N_CORRECT = 100
N_INCORRECT = 100
N_LAYERS = 12  # simulate a 12-layer model (small enough for fast CPU run)
SEQ_LEN = 32  # tokens per sequence
HIDDEN_DIM = 64  # hidden dimension per token
TRAIN_FRAC = 0.80  # 80% train, 20% eval
RANDOM_SEED = 42

# Exp 923 baseline AUC (uniform-weight ensemble, tier0i_no_improvement)
BASELINE_AUC_923 = 0.5625


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _make_hidden_states(
    n_samples: int,
    n_layers: int,
    seq_len: int,
    hidden_dim: int,
    inject_drift: bool,
    rng: np.random.Generator,
) -> list:
    """Generate synthetic per-layer hidden states for one class.

    WHY SYNTHETIC:
        We need controlled ground truth: "incorrect" responses should have a
        distinctive drift pattern that a learned probe can pick up.  In real
        models this pattern is latent and partially dataset-dependent; here we
        simulate it explicitly so we can measure ceiling performance for the
        architectural change before applying it to real models.

    For CORRECT responses: small, uniform Gaussian noise on each layer —
        hidden states vary slowly token-to-token (low drift everywhere).

    For INCORRECT responses: large noise injected on alternating layers (0, 2, 4, …)
        to simulate the non-local, layer-skipping drift pattern described in
        arXiv 2604.17121.  Even layers are "drift-heavy", odd layers are "quiet".
        A uniform-weight probe averages these together and gets ~0.50 AUC.
        A learned probe can upweight the even layers and get much higher AUC.

    Args:
        n_samples: number of samples to generate.
        n_layers: number of transformer layers.
        seq_len: tokens per sequence.
        hidden_dim: hidden dimension per token.
        inject_drift: True for incorrect class (large alternating-layer noise).
        rng: seeded random generator for reproducibility.

    Returns:
        list of n_samples; each sample is a list of n_layers NDArrays
        shape [seq_len, hidden_dim].
    """
    samples = []
    for _ in range(n_samples):
        layers = []
        # Base activation: smoothly varying sequence (low drift by construction).
        base = rng.normal(0, 0.1, (seq_len, hidden_dim)).astype(np.float32)
        for i in range(n_layers):
            if inject_drift and (i % 2 == 0):
                # Large additive noise on even layers — each token independently
                # perturbed, destroying local smoothness and causing high drift.
                noise = rng.normal(0, 1.5, (seq_len, hidden_dim)).astype(np.float32)
            else:
                noise = rng.normal(0, 0.05, (seq_len, hidden_dim)).astype(np.float32)
            layers.append(base + noise)
        samples.append(layers)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Execute the depth-recurrent probe training + evaluation pipeline.

    Returns the result dict to be passed to tmpl.build_result().
    """
    rng = np.random.default_rng(RANDOM_SEED)

    with tmpl.phase("generate_synthetic_data"):
        correct_hs = _make_hidden_states(
            N_CORRECT, N_LAYERS, SEQ_LEN, HIDDEN_DIM, inject_drift=False, rng=rng
        )
        incorrect_hs = _make_hidden_states(
            N_INCORRECT, N_LAYERS, SEQ_LEN, HIDDEN_DIM, inject_drift=True, rng=rng
        )

        # Combine and create labels: 0=correct, 1=incorrect.
        all_hs = correct_hs + incorrect_hs
        all_labels = [0] * N_CORRECT + [1] * N_INCORRECT

        # Shuffle together (deterministic via rng).
        indices = np.arange(len(all_hs))
        rng.shuffle(indices)
        all_hs = [all_hs[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

        n_total = len(all_hs)
        n_train = int(n_total * TRAIN_FRAC)
        X_train = all_hs[:n_train]
        y_train = all_labels[:n_train]
        X_eval = all_hs[n_train:]
        y_eval = all_labels[n_train:]

    with tmpl.phase("train_probe"):
        probe = DRIFTProbeV3(hidden_dim=32, lr=0.05, n_iter=500)
        probe.fit(X_train, y_train)
        attn_weights = probe.layer_attention_weights().tolist()

    with tmpl.phase("evaluate_probe"):
        proba = probe.predict_proba(X_eval)
        probe_auc = float(roc_auc_score(y_eval, proba))

    # Determine honest verdict.
    if probe_auc > BASELINE_AUC_923 and probe_auc > 0.50:
        honest_verdict = "depth_recurrent_improves"
    elif probe_auc > 0.50:
        honest_verdict = "depth_recurrent_marginal"
    else:
        honest_verdict = "depth_recurrent_no_improvement"

    delta_auc = round(probe_auc - BASELINE_AUC_923, 4)

    return {
        "probe_auc": round(probe_auc, 4),
        "baseline_auc_923": BASELINE_AUC_923,
        "delta_auc": delta_auc,
        "honest_verdict": honest_verdict,
        "n_layers": N_LAYERS,
        "n_train": n_train,
        "n_eval": len(X_eval),
        "n_correct_total": N_CORRECT,
        "n_incorrect_total": N_INCORRECT,
        "layer_attention_weights": attn_weights,
        "inference_mode": "synthetic_cpu",
        "decision_class": "detect",
    }


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    result_data = run_experiment(tmpl)

    artifact = tmpl.build_result(result_data, status="success")

    with open(DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)

    print(
        f"probe_auc={result_data['probe_auc']}  baseline={BASELINE_AUC_923}"
        f"  delta={result_data['delta_auc']}  verdict={result_data['honest_verdict']}"
    )
    print(f"Deliverable written: {DELIVERABLE}")


if __name__ == "__main__":
    main()
