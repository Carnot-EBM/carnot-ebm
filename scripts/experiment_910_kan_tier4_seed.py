#!/usr/bin/env python3
"""Exp 910: KAN Tier 4 Seed — AutoKnots adaptive grid refinement on GSM8K verification.

**Researcher summary:**
    FR-11 Tier 4 (KAN adaptive structure) has NOT STARTED.  This experiment
    seeds it by implementing and measuring the AutoKnots technique from
    arXiv 2412.13423: add knots to high-activation splines, remove from
    dormant ones.  We measure whether this structural self-improvement raises
    AUC on a synthetic GSM8K verification task.

**What this experiment does:**
    1. Creates a small KANModel (CPU, no GPU required) configured for binary
       classification of correct vs. incorrect math answer embeddings.
    2. Generates 50 synthetic GSM8K-style (question, response) binary feature
       vectors: 25 "correct" (low energy) and 25 "wrong" (high energy).
    3. Measures baseline AUC using Wilcoxon-Mann-Whitney U.
    4. Runs AutoKnotsRefiner.multi_round_refine(activations, rounds=3).
    5. Measures post-refinement AUC and reports the signed improvement.

**Why synthetic embeddings:**
    This is a CPU seed experiment (no GGUF models).  The embeddings simulate
    the structure of real GSM8K binary features (digit-presence flags, sign
    flags, operator flags) with a small ground-truth separation baked in.
    Real GGUF inference is deferred to the full Tier 4 implementation once
    the structural approach is validated here.

**Verdict:**
    "tier4_seed_viable"        if post_refinement_auc > baseline_auc
    "tier4_seed_no_improvement" otherwise

Spec: REQ-SELF-008, SCENARIO-SELF-008
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so imports work when run from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.models.kan import KANConfig, KANModel  # noqa: E402
from carnot.models.kan_autoknots import AutoKnotsRefiner  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 910
TITLE = "KAN Tier 4 Seed — AutoKnots Adaptive Grid Refinement"
DELIVERABLE = "results/experiment_910_kan_tier4_seed.json"

INPUT_DIM = 16       # Binary feature vector: digit flags, sign flags, operator flags
NUM_KNOTS = 8        # Starting knot count — mid-range so both add and remove can fire
N_SAMPLES = 50       # 25 correct + 25 wrong (GSM8K verification task)
SEED = 42
REFINEMENT_ROUNDS = 3

# Thresholds tuned for binary {0,1} inputs where expected magnitudes are 0–1.
HIGH_THRESH = 0.45   # Above this → add knot (spline is active)
LOW_THRESH = 0.05    # Below this → remove knot (spline is dormant)


def _make_gsm8k_embeddings(
    n_samples: int, input_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic binary embeddings simulating GSM8K verification features.

    The first half of the feature vector (bits 0..7) encodes "correct answer"
    structure (digits appear in the right places).  The second half (bits 8..15)
    encodes "wrong answer" noise.

    Correct responses: bits 0..7 all set, bits 8..15 sparse.
    Wrong responses: bits 0..7 sparse, bits 8..15 mixed.

    This creates a linearly separable embedding with realistic sparsity, which
    is what a trained KAN should be able to discriminate.

    Returns:
        Tuple of (correct_embeddings, wrong_embeddings, full_batch).
        full_batch is correct then wrong, shape (n_samples, input_dim).
        Labels: 0=correct, 1=wrong.
    """
    half = n_samples // 2
    rng = np.random.default_rng(seed)

    # Correct: first half of dims active with high probability
    correct = rng.binomial(1, 0.8, (half, input_dim // 2)).astype(np.float32)
    correct_noise = rng.binomial(1, 0.1, (half, input_dim // 2)).astype(np.float32)
    correct_embs = np.concatenate([correct, correct_noise], axis=1)

    # Wrong: first half sparse, second half noisy
    wrong_signal = rng.binomial(1, 0.2, (half, input_dim // 2)).astype(np.float32)
    wrong_noise = rng.binomial(1, 0.5, (half, input_dim // 2)).astype(np.float32)
    wrong_embs = np.concatenate([wrong_signal, wrong_noise], axis=1)

    full_batch = np.concatenate([correct_embs, wrong_embs], axis=0)
    return correct_embs, wrong_embs, full_batch


def _compute_auc(energies_correct: np.ndarray, energies_wrong: np.ndarray) -> float:
    """Compute AUROC via Wilcoxon-Mann-Whitney U statistic.

    The KAN is an energy function: LOWER energy = more likely correct.
    AUC = P(E_correct < E_wrong).  A perfect energy model has AUC = 1.0.

    This is computed exactly (not via sklearn) to avoid adding a dependency
    and to keep the calculation transparent for the retro audit.

    Args:
        energies_correct: Energy values for correct responses, shape (n_c,).
        energies_wrong: Energy values for wrong responses, shape (n_w,).

    Returns:
        AUROC in [0, 1].
    """
    wins = 0
    pairs = 0
    for ec in energies_correct:
        for ew in energies_wrong:
            pairs += 1
            if ec < ew:
                wins += 1
            elif ec == ew:
                wins += 0.5
    return wins / max(pairs, 1)


def _run_kan_eval(
    kan: KANModel, correct_embs: np.ndarray, wrong_embs: np.ndarray
) -> float:
    """Evaluate KAN AUC on correct vs. wrong embeddings.

    Args:
        kan: KANModel to evaluate.
        correct_embs: shape (n, input_dim).
        wrong_embs: shape (n, input_dim).

    Returns:
        AUROC float.
    """
    import jax.numpy as jnp

    e_correct = np.array([
        float(kan.energy(jnp.array(x, dtype=jnp.float32)))
        for x in correct_embs
    ])
    e_wrong = np.array([
        float(kan.energy(jnp.array(x, dtype=jnp.float32)))
        for x in wrong_embs
    ])
    return _compute_auc(e_correct, e_wrong)


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # -----------------------------------------------------------------------
    # Build KANModel (small, CPU)
    # -----------------------------------------------------------------------
    import jax.random as jrandom

    config = KANConfig(
        input_dim=INPUT_DIM,
        num_knots=NUM_KNOTS,
        degree=3,
        sparse=False,     # fully connected for a 16-dim model (120 edges)
    )
    kan = KANModel(config, key=jrandom.PRNGKey(SEED))

    # -----------------------------------------------------------------------
    # Synthetic GSM8K embeddings
    # -----------------------------------------------------------------------
    correct_embs, wrong_embs, full_batch = _make_gsm8k_embeddings(
        N_SAMPLES, INPUT_DIM, SEED
    )

    # -----------------------------------------------------------------------
    # Baseline AUC (untrained KAN — random control points)
    # -----------------------------------------------------------------------
    baseline_auc = _run_kan_eval(kan, correct_embs, wrong_embs)

    # -----------------------------------------------------------------------
    # AutoKnots refinement
    # -----------------------------------------------------------------------
    refiner = AutoKnotsRefiner(
        kan_model=kan,
        high_activation_threshold=HIGH_THRESH,
        low_activation_threshold=LOW_THRESH,
        max_knots_per_spline=32,
        min_knots_per_spline=4,
    )
    refinement_results = refiner.multi_round_refine(full_batch, rounds=REFINEMENT_ROUNDS)

    total_added = sum(r.n_added for r in refinement_results)
    total_removed = sum(r.n_removed for r in refinement_results)

    # -----------------------------------------------------------------------
    # Post-refinement AUC
    # -----------------------------------------------------------------------
    post_refinement_auc = _run_kan_eval(kan, correct_embs, wrong_embs)

    signed_auc_improvement = post_refinement_auc - baseline_auc

    if post_refinement_auc > baseline_auc:
        honest_verdict = "tier4_seed_viable"
    else:
        honest_verdict = "tier4_seed_no_improvement"

    # -----------------------------------------------------------------------
    # Build artifact
    # -----------------------------------------------------------------------
    round_summaries = [
        {
            "round": idx + 1,
            "n_added": r.n_added,
            "n_removed": r.n_removed,
            "n_splines_modified": len(r.splines_modified),
        }
        for idx, r in enumerate(refinement_results)
    ]

    payload = {
        "honest_verdict": honest_verdict,
        "baseline_auc": round(baseline_auc, 6),
        "post_refinement_auc": round(post_refinement_auc, 6),
        "signed_auc_improvement": round(signed_auc_improvement, 6),
        "n_knots_added": total_added,
        "n_knots_removed": total_removed,
        "net_knots_change": total_added - total_removed,
        "refinement_rounds": REFINEMENT_ROUNDS,
        "round_summaries": round_summaries,
        "n_gsm8k_samples": N_SAMPLES,
        "input_dim": INPUT_DIM,
        "num_knots_initial": NUM_KNOTS,
        "high_activation_threshold": HIGH_THRESH,
        "low_activation_threshold": LOW_THRESH,
        "kan_config": {
            "input_dim": INPUT_DIM,
            "num_knots": NUM_KNOTS,
            "degree": 3,
            "sparse": False,
        },
        "models_used": ["synthetic_embeddings_cpu_only"],
        "tier": "FR-11 Tier 4",
    }

    artifact = tmpl.build_result(payload, status="success")

    output_path = Path(DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"honest_verdict:         {honest_verdict}")
    print(f"baseline_auc:           {baseline_auc:.4f}")
    print(f"post_refinement_auc:    {post_refinement_auc:.4f}")
    print(f"signed_auc_improvement: {signed_auc_improvement:+.4f}")
    print(f"knots added:            {total_added}")
    print(f"knots removed:          {total_removed}")
    print(f"Deliverable written: {output_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
