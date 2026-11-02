#!/usr/bin/env python3
"""Exp 936: KAN Tier 4 Real Data — AutoKnots refinement on real FoVer-labeled violation pairs.

**Researcher summary:**
    Exp 910 validated AutoKnots adaptive knot refinement on SYNTHETIC GSM8K embeddings
    and produced honest_verdict='tier4_seed_viable' with AUC improvement +0.0624.
    This experiment applies the SAME AutoKnots approach to REAL FoVer-labeled violation
    pairs accumulated from live GPU inference (Exp 905 and prior sessions).

    The question: does training on real human-labeled violation data (as opposed to
    synthetic embeddings) yield higher or comparable post-refinement AUC?

**What this experiment does:**
    1. Loads real FoVer-labeled data from results/fover_labeled_steps_live.json.
       - If >= 20 labeled pairs: use them (inference_mode='real_fover_data').
       - If < 20 labeled pairs: fall back to synthetic (inference_mode='synthetic_fallback').
    2. Converts text pairs to 16-dim binary feature vectors using lexical feature
       extraction (digit/operator/sign presence flags — same feature schema as Exp 910).
    3. Splits 80/20 train/held-out.
    4. Trains a KANModel (CPU) on training split.
    5. Measures baseline AUC on held-out split (pre-refinement).
    6. Runs AutoKnotsRefiner.multi_round_refine on training set activations.
    7. Measures post-refinement AUC on held-out split.
    8. Compares against Exp 910 synthetic baseline AUC (0.1584 pre / 0.2208 post).

**honest_verdict:**
    'real_data_improves_over_synthetic'       if real post_auc > exp910 post_auc (0.2208)
    'real_data_comparable'                    if |real_post_auc - exp910_post_auc| < 0.05
    'real_data_insufficient_synthetic_fallback' if n_real < 20

**Prior experiment:**
    Exp 910: tier4_seed_viable, baseline_auc=0.1584, post_refinement_auc=0.2208.
    Exp 925: blocked_missing_prior_failures (no code change needed).

Spec: REQ-SELF-008, SCENARIO-SELF-008
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.models.kan import KANConfig, KANModel  # noqa: E402
from carnot.models.kan_autoknots import AutoKnotsRefiner  # noqa: E402

EXP_ID = 936
TITLE = "KAN Tier 4 Real Data — AutoKnots on FoVer-Labeled Violation Pairs"
DELIVERABLE = "results/experiment_936_kan_tier4_real_data.json"

FOVER_DATA_PATH = "results/fover_labeled_steps_live.json"
MIN_REAL_PAIRS = 20

INPUT_DIM = 16
NUM_KNOTS = 8
SEED = 42
REFINEMENT_ROUNDS = 3
TRAIN_FRAC = 0.8

HIGH_THRESH = 0.45
LOW_THRESH = 0.05

# Exp 910 synthetic baseline for comparison
EXP910_BASELINE_AUC = 0.1584
EXP910_POST_AUC = 0.2208


def _extract_features(text: str) -> np.ndarray:
    """Extract 16-dim binary feature vector from a step text string.

    Features are chosen to mirror the synthetic GSM8K embedding schema from Exp 910:
    - Bits 0-7: "correct answer" structural features (digit presence, arithmetic operators,
      conclusion keywords, equation markers, unit references, result patterns, equality,
      decimal presence).
    - Bits 8-15: "wrong answer" noise features (negation words, conditional hedging,
      excessive question marks, partial answer markers, contradiction patterns,
      no-number spans, long run-on text, missing conclusion).

    This uses only lexical regex features — no LLM inference — so it works fully CPU-local.

    Args:
        text: Raw step text from a FoVer annotation pair.

    Returns:
        Binary float32 feature vector of shape (16,).
    """
    t = text.lower()
    feats = np.zeros(16, dtype=np.float32)

    # Bits 0-7: correct-answer structural indicators
    feats[0] = float(bool(re.search(r"\d+", t)))  # any digit present
    feats[1] = float(bool(re.search(r"[+\-*/÷×]", t)))  # arithmetic operator
    feats[2] = float(bool(re.search(r"\btherefore\b|\bthus\b|\bso\b", t)))  # conclusion word
    feats[3] = float(bool(re.search(r"=\s*\d", t)))  # equation with result
    feats[4] = float(bool(re.search(r"\b(kg|km|m|cm|hrs?|min|dollars?|\$|%)\b", t)))  # units
    feats[5] = float(bool(re.search(r"\btotal\b|\bresult\b|\banswer\b", t)))  # result word
    feats[6] = float(bool(re.search(r"\d+\s*=\s*\d+", t)))  # numeric equality
    feats[7] = float(bool(re.search(r"\d+\.\d+", t)))  # decimal number

    # Bits 8-15: wrong-answer noise indicators
    feats[8] = float(bool(re.search(r"\bnot\b|\bno\b|\bnever\b|\bcannot\b", t)))  # negation
    feats[9] = float(bool(re.search(r"\bif\b|\bmight\b|\bcould\b|\bmaybe\b", t)))  # hedging
    feats[10] = float(t.count("?") > 1)  # multiple question marks
    feats[11] = float(bool(re.search(r"\bpartially\b|\bincomplete\b|\bunfinished\b", t)))
    feats[12] = float(
        bool(re.search(r"\bbut\b.*\bhowever\b|\bhowever\b.*\bbut\b", t))
    )  # contradiction
    feats[13] = float(not bool(re.search(r"\d", t)))  # no digits at all
    feats[14] = float(len(t) > 500)  # very long run-on
    feats[15] = float(
        not bool(re.search(r"\btherefore\b|\bthus\b|\btotal\b|\banswer\b", t))
    )  # no conclusion

    return feats


def _load_real_fover_pairs(path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load and featurize real FoVer-labeled step texts.

    Reads JSON from `path`. Expected schema: list of dicts with keys
    'step_text' and 'label' (values 'correct' or 'incorrect').

    Args:
        path: File path to the FoVer labeled JSON.

    Returns:
        Tuple (correct_embs, incorrect_embs), each a list of float32 arrays shape (16,).
    """
    with open(path) as f:
        pairs = json.load(f)

    correct_embs = []
    incorrect_embs = []
    for p in pairs:
        feats = _extract_features(p["step_text"])
        label = p.get("label", "")
        if label == "correct":
            correct_embs.append(feats)
        elif label in ("incorrect", "wrong"):
            incorrect_embs.append(feats)

    return correct_embs, incorrect_embs


def _make_synthetic_fallback(
    n_samples: int, input_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic embeddings as fallback when real data is insufficient.

    Replicates Exp 910 synthetic generation exactly so results are comparable.

    Args:
        n_samples: Total samples (half correct, half wrong).
        input_dim: Feature vector dimension.
        seed: RNG seed.

    Returns:
        Tuple (correct_embs, wrong_embs), each shape (n_samples//2, input_dim).
    """
    half = n_samples // 2
    rng = np.random.default_rng(seed)

    correct = rng.binomial(1, 0.8, (half, input_dim // 2)).astype(np.float32)
    correct_noise = rng.binomial(1, 0.1, (half, input_dim // 2)).astype(np.float32)
    correct_embs = np.concatenate([correct, correct_noise], axis=1)

    wrong_signal = rng.binomial(1, 0.2, (half, input_dim // 2)).astype(np.float32)
    wrong_noise = rng.binomial(1, 0.5, (half, input_dim // 2)).astype(np.float32)
    wrong_embs = np.concatenate([wrong_signal, wrong_noise], axis=1)

    return correct_embs, wrong_embs


def _compute_auc(energies_correct: np.ndarray, energies_wrong: np.ndarray) -> float:
    """Compute AUROC via Wilcoxon-Mann-Whitney U statistic.

    Lower energy = model thinks input is correct. AUC = P(E_correct < E_wrong).
    Perfect model: AUC = 1.0. Random: AUC = 0.5.

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


def _run_kan_eval(kan: KANModel, correct_embs: np.ndarray, wrong_embs: np.ndarray) -> float:
    """Evaluate KAN AUC on correct vs. wrong embedding arrays.

    Args:
        kan: Trained KANModel.
        correct_embs: shape (n, input_dim).
        wrong_embs: shape (n, input_dim).

    Returns:
        AUROC float.
    """
    import jax.numpy as jnp

    e_correct = np.array([float(kan.energy(jnp.array(x, dtype=jnp.float32))) for x in correct_embs])
    e_wrong = np.array([float(kan.energy(jnp.array(x, dtype=jnp.float32))) for x in wrong_embs])
    return _compute_auc(e_correct, e_wrong)


def _train_split(
    correct_embs: np.ndarray,
    wrong_embs: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split correct and wrong embeddings into train and held-out sets.

    Maintains class balance: train_frac of each class goes to train.

    Args:
        correct_embs: All correct embeddings, shape (n_c, d).
        wrong_embs: All wrong embeddings, shape (n_w, d).
        train_frac: Fraction for training (e.g., 0.8).
        seed: RNG seed for shuffle.

    Returns:
        Tuple (correct_train, correct_test, wrong_train, wrong_test).
    """
    rng = np.random.default_rng(seed)

    idx_c = rng.permutation(len(correct_embs))
    n_train_c = max(1, int(len(correct_embs) * train_frac))
    c_train = correct_embs[idx_c[:n_train_c]]
    c_test = correct_embs[idx_c[n_train_c:]]

    idx_w = rng.permutation(len(wrong_embs))
    n_train_w = max(1, int(len(wrong_embs) * train_frac))
    w_train = wrong_embs[idx_w[:n_train_w]]
    w_test = wrong_embs[idx_w[n_train_w:]]

    return c_train, c_test, w_train, w_test


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    import jax.random as jrandom

    # -----------------------------------------------------------------------
    # Load data: real FoVer pairs or synthetic fallback
    # -----------------------------------------------------------------------
    fover_path = Path(FOVER_DATA_PATH)
    n_real = 0
    inference_mode = "synthetic_fallback"

    if fover_path.exists():
        correct_list, incorrect_list = _load_real_fover_pairs(str(fover_path))
        n_real = len(correct_list) + len(incorrect_list)

    if n_real >= MIN_REAL_PAIRS and correct_list and incorrect_list:
        inference_mode = "real_fover_data"
        correct_embs = np.stack(correct_list, axis=0)
        wrong_embs = np.stack(incorrect_list, axis=0)
        print(
            f"Using real FoVer data: {len(correct_list)} correct, {len(incorrect_list)} incorrect"
        )
    else:
        print(f"Real pairs={n_real} < {MIN_REAL_PAIRS}, using synthetic fallback")
        correct_embs, wrong_embs = _make_synthetic_fallback(50, INPUT_DIM, SEED)

    # -----------------------------------------------------------------------
    # 80/20 train/test split
    # -----------------------------------------------------------------------
    c_train, c_test, w_train, w_test = _train_split(correct_embs, wrong_embs, TRAIN_FRAC, SEED)

    # -----------------------------------------------------------------------
    # Build KANModel (small, CPU, same config as Exp 910)
    # -----------------------------------------------------------------------
    config = KANConfig(
        input_dim=INPUT_DIM,
        num_knots=NUM_KNOTS,
        degree=3,
        sparse=False,
    )
    kan = KANModel(config, key=jrandom.PRNGKey(SEED))

    # -----------------------------------------------------------------------
    # Train on training split using contrastive divergence
    # Labels: correct=0 (want low energy), wrong=1 (want high energy)
    # train_cd expects a data matrix; we pass both classes in full_batch
    # -----------------------------------------------------------------------
    full_train = np.concatenate([c_train, w_train], axis=0)
    kan.train_cd(full_train, n_epochs=20, lr=0.01)

    # -----------------------------------------------------------------------
    # Baseline AUC on held-out test set (post-training, pre-refinement)
    # -----------------------------------------------------------------------
    if len(c_test) == 0 or len(w_test) == 0:
        # If split produced empty test set (very small data), use full set
        c_test = correct_embs
        w_test = wrong_embs

    baseline_auc = _run_kan_eval(kan, c_test, w_test)

    # -----------------------------------------------------------------------
    # Activation histogram on training set, then AutoKnots refinement
    # -----------------------------------------------------------------------
    refiner = AutoKnotsRefiner(
        kan_model=kan,
        high_activation_threshold=HIGH_THRESH,
        low_activation_threshold=LOW_THRESH,
        max_knots_per_spline=32,
        min_knots_per_spline=4,
    )
    refinement_results = refiner.multi_round_refine(full_train, rounds=REFINEMENT_ROUNDS)

    total_added = sum(r.n_added for r in refinement_results)
    total_removed = sum(r.n_removed for r in refinement_results)

    # -----------------------------------------------------------------------
    # Post-refinement AUC on held-out set
    # -----------------------------------------------------------------------
    post_refinement_auc = _run_kan_eval(kan, c_test, w_test)
    signed_auc_improvement = post_refinement_auc - baseline_auc

    # -----------------------------------------------------------------------
    # honest_verdict vs Exp 910 synthetic baseline
    # -----------------------------------------------------------------------
    if inference_mode == "synthetic_fallback":
        honest_verdict = "real_data_insufficient_synthetic_fallback"
    elif post_refinement_auc > EXP910_POST_AUC:
        honest_verdict = "real_data_improves_over_synthetic"
    elif abs(post_refinement_auc - EXP910_POST_AUC) < 0.05:
        honest_verdict = "real_data_comparable"
    else:
        honest_verdict = "real_data_below_synthetic"

    # -----------------------------------------------------------------------
    # Artifact
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
        "inference_mode": inference_mode,
        "n_real_pairs": n_real,
        "n_correct_total": len(correct_embs),
        "n_wrong_total": len(wrong_embs),
        "n_train_correct": len(c_train),
        "n_train_wrong": len(w_train),
        "n_test_correct": len(c_test),
        "n_test_wrong": len(w_test),
        "baseline_auc": round(baseline_auc, 6),
        "post_refinement_auc": round(post_refinement_auc, 6),
        "signed_auc_improvement": round(signed_auc_improvement, 6),
        "exp910_baseline_auc": EXP910_BASELINE_AUC,
        "exp910_post_refinement_auc": EXP910_POST_AUC,
        "delta_vs_exp910_post": round(post_refinement_auc - EXP910_POST_AUC, 6),
        "n_knots_added": total_added,
        "n_knots_removed": total_removed,
        "net_knots_change": total_added - total_removed,
        "refinement_rounds": REFINEMENT_ROUNDS,
        "round_summaries": round_summaries,
        "input_dim": INPUT_DIM,
        "num_knots_initial": NUM_KNOTS,
        "high_activation_threshold": HIGH_THRESH,
        "low_activation_threshold": LOW_THRESH,
        "train_frac": TRAIN_FRAC,
        "kan_config": {
            "input_dim": INPUT_DIM,
            "num_knots": NUM_KNOTS,
            "degree": 3,
            "sparse": False,
        },
        "models_used": ["cpu_only_lexical_features"],
        "tier": "FR-11 Tier 4",
        "prior_experiment": "exp910_kan_tier4_seed_auc_0.2208",
    }

    artifact = tmpl.build_result(payload, status="success")

    output_path = Path(DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"honest_verdict:         {honest_verdict}")
    print(f"inference_mode:         {inference_mode}")
    print(f"n_real_pairs:           {n_real}")
    print(f"baseline_auc:           {baseline_auc:.4f}")
    print(f"post_refinement_auc:    {post_refinement_auc:.4f}")
    print(f"signed_auc_improvement: {signed_auc_improvement:+.4f}")
    print(f"delta_vs_exp910_post:   {post_refinement_auc - EXP910_POST_AUC:+.4f}")
    print(f"knots added:            {total_added}")
    print(f"knots removed:          {total_removed}")
    print(f"Deliverable written: {output_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
