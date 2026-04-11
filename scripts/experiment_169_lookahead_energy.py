#!/usr/bin/env python3
"""Experiment 169 — Lookahead Energy: TruthfulQA AUROC Benchmark.

**Researcher summary:**
    Benchmarks LookaheadEnergyExtractor on 50 simulated TruthfulQA-style
    factual questions. "Correct" answers receive peaked logit distributions
    (high confidence = low NLL = low lookahead energy); "incorrect" answers
    receive flatter logit distributions (uncertainty = higher NLL = higher
    lookahead energy). Measures AUROC for: spilled energy alone, lookahead
    energy alone, and combined (max of both). The "combined" signal tests
    whether arxiv 2512.15605 and arxiv 2602.18671 are complementary.

    Target: lookahead AUROC > 0.60, combined AUROC ≥ spilled_only AUROC.

**Background:**
    arxiv 2512.15605 (2025) establishes that autoregressive LLMs are
    implicitly EBMs: the "lookahead energy" of a response equals
    -log P(response | prompt) = sum(-log p(token_t)), i.e., the mean
    per-token NLL computed during generation. This is distinct from
    "spilled energy" (arxiv 2602.18671):

        Spilled energy   = max(0, NLL_argmax − entropy(p))
                         Measures how much probability was "wasted" across the
                         vocabulary (high when model is uncertain between tokens).

        Lookahead energy = NLL_argmax  (= per-token cross-entropy loss)
                         Measures the absolute likelihood of the generated
                         response under the model (high when model is
                         "surprised" by its own output).

    Both signals are KB-free and computed directly from generation logits.
    This experiment tests whether combining them (via max) gives a better
    hallucination detector than either alone.

**Methodology:**
    Same simulation approach as Exp 157:
    - Correct answers: logits with strong peak (one high-logit token per
      position). Low entropy → low NLL → low lookahead energy.
    - Wrong answers: logits drawn from N(0, noise_std). Near-uniform
      distribution → high NLL → high lookahead energy.

Run:
    JAX_PLATFORMS=cpu python scripts/experiment_169_lookahead_energy.py

Output:
    Prints AUROC for spilled, lookahead, and combined signals.
    Saves results to results/experiment_169_results.json.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.pipeline.spilled_energy import (
    DEFAULT_SPILLED_THRESHOLD,
    SpilledEnergyExtractor,
)
from carnot.pipeline.lookahead_energy import (
    DEFAULT_LOOKAHEAD_THRESHOLD,
    LookaheadEnergyExtractor,
)

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

N_QUESTIONS = 50        # TruthfulQA subset size
N_CORRECT = 25          # Half are "correct" (low energy)
N_WRONG = 25            # Half are "incorrect" (high energy)
VOCAB_SIZE = 1000       # Smaller vocab for fast CPU simulation
N_TOKENS = 20           # Tokens per generated answer
CORRECT_PEAK_LOGIT = 8.0    # Logit advantage for the argmax token (correct)
WRONG_NOISE_STD = 0.5       # Std of Gaussian noise for uncertain logits (wrong)
SEED = 169

# Threshold sweep for precision/recall analysis
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Sample TruthfulQA-style questions for the report.
SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What year did World War II end?",
    "What is the chemical symbol for water?",
    "Who wrote Hamlet?",
    "What planet is closest to the sun?",
    "What is the speed of light in vacuum?",
    "What is the boiling point of water at sea level?",
    "Who painted the Mona Lisa?",
    "What is the largest organ in the human body?",
]


# ---------------------------------------------------------------------------
# Logit simulation helpers
# ---------------------------------------------------------------------------


def make_correct_logits(
    key: jax.Array,
    n_tokens: int,
    vocab_size: int,
    peak_logit: float,
) -> jnp.ndarray:
    """Simulate logits for a confident (correct) answer.

    **Detailed explanation for engineers:**
        One token index is chosen as the "answer token" at each position.
        That token gets a high logit value; all others get 0. The resulting
        softmax is strongly peaked → low NLL → low lookahead energy.

    Args:
        key: JAX PRNG key.
        n_tokens: Number of generated tokens.
        vocab_size: Vocabulary size.
        peak_logit: Logit value for the chosen token (e.g., 8.0).

    Returns:
        Logits of shape (n_tokens, vocab_size).
    """
    key, subkey = jrandom.split(key)
    logits = jnp.zeros((n_tokens, vocab_size))
    chosen = jrandom.randint(subkey, shape=(n_tokens,), minval=0, maxval=vocab_size)
    logits = logits.at[jnp.arange(n_tokens), chosen].set(peak_logit)
    return logits


def make_wrong_logits(
    key: jax.Array,
    n_tokens: int,
    vocab_size: int,
    noise_std: float,
) -> jnp.ndarray:
    """Simulate logits for an uncertain (wrong/hallucinated) answer.

    **Detailed explanation for engineers:**
        Logits are drawn from N(0, noise_std) — a nearly flat distribution
        without any dominant token. The resulting softmax is spread across
        many tokens → higher NLL → higher lookahead energy than correct logits.

    Args:
        key: JAX PRNG key.
        n_tokens: Number of generated tokens.
        vocab_size: Vocabulary size.
        noise_std: Standard deviation of the Gaussian noise.

    Returns:
        Logits of shape (n_tokens, vocab_size).
    """
    return jrandom.normal(key, shape=(n_tokens, vocab_size)) * noise_std


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC (Area Under the ROC Curve) via trapezoidal rule.

    **Detailed explanation for engineers:**
        AUROC = probability that a randomly chosen positive example has a
        higher score than a randomly chosen negative example. Here:
        - label=1 → "wrong" (high energy expected → positive class)
        - label=0 → "correct" (low energy expected → negative class)
        - score  → energy value (higher = more likely hallucination)

    Args:
        labels: Binary labels (1 = hallucination/wrong, 0 = correct).
        scores: Energy values (higher = more likely hallucination).

    Returns:
        AUROC in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case

    # Sort by score descending (high score = predicted positive).
    paired = sorted(zip(scores, labels), reverse=True)

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = 0
    fp = 0

    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    # Trapezoidal integration (FPR on x-axis, TPR on y-axis).
    auroc = 0.0
    for i in range(1, len(fpr_points)):
        dx = fpr_points[i] - fpr_points[i - 1]
        avg_y = (tpr_points[i] + tpr_points[i - 1]) / 2.0
        auroc += dx * avg_y

    return auroc


def compute_precision_recall(
    labels: list[int], scores: list[float], threshold: float
) -> tuple[float, float]:
    """Precision and recall at a given threshold.

    Args:
        labels: Binary labels (1 = wrong/hallucination).
        scores: Energy values.
        threshold: Predict positive (hallucination) when score > threshold.

    Returns:
        (precision, recall) tuple.
    """
    tp = sum(1 for la, s in zip(labels, scores) if la == 1 and s > threshold)
    fp = sum(1 for la, s in zip(labels, scores) if la == 0 and s > threshold)
    fn = sum(1 for la, s in zip(labels, scores) if la == 1 and s <= threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict:
    """Run the Exp 169 benchmark and return a results dictionary.

    Returns:
        Dictionary with AUROC for spilled, lookahead, combined, and per-sample data.
    """
    key = jrandom.PRNGKey(SEED)
    spilled_extractor = SpilledEnergyExtractor(threshold=DEFAULT_SPILLED_THRESHOLD)
    lookahead_extractor = LookaheadEnergyExtractor(threshold=DEFAULT_LOOKAHEAD_THRESHOLD)

    labels: list[int] = []
    spilled_scores: list[float] = []
    lookahead_scores: list[float] = []
    per_sample: list[dict] = []

    print("Exp 169 — Lookahead Energy: TruthfulQA AUROC Benchmark")
    print(f"  N={N_QUESTIONS} simulated TruthfulQA questions")
    print(f"  Vocab={VOCAB_SIZE}, tokens/answer={N_TOKENS}")
    print(f"  Correct logit peak={CORRECT_PEAK_LOGIT}, wrong noise_std={WRONG_NOISE_STD}")
    print()

    # --- Generate simulated logits for each question ---
    for i in range(N_QUESTIONS):
        key, subkey = jrandom.split(key)
        is_wrong = i >= N_CORRECT  # First half correct, second half wrong

        if is_wrong:
            logits = make_wrong_logits(subkey, N_TOKENS, VOCAB_SIZE, WRONG_NOISE_STD)
            label = 1
            answer_type = "wrong"
        else:
            logits = make_correct_logits(subkey, N_TOKENS, VOCAB_SIZE, CORRECT_PEAK_LOGIT)
            label = 0
            answer_type = "correct"

        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)]

        # Spilled energy signal.
        spilled_results = spilled_extractor.extract(question, logits=logits)
        spilled = spilled_results[0].metadata["spilled_energy"] if spilled_results else 0.0
        spilled_satisfied = spilled_results[0].metadata["satisfied"] if spilled_results else True

        # Lookahead energy signal.
        lookahead_results = lookahead_extractor.extract(question, logits=logits)
        lookahead = lookahead_results[0].metadata["lookahead_energy"] if lookahead_results else 0.0
        lookahead_satisfied = (
            lookahead_results[0].metadata["satisfied"] if lookahead_results else True
        )

        labels.append(label)
        spilled_scores.append(spilled)
        lookahead_scores.append(lookahead)
        per_sample.append({
            "index": i,
            "question": question,
            "answer_type": answer_type,
            "label": label,
            "spilled_energy": spilled,
            "lookahead_energy": lookahead,
            "spilled_satisfied": spilled_satisfied,
            "lookahead_satisfied": lookahead_satisfied,
        })

    # Combined signal: max of spilled and lookahead (normalized to comparable scale).
    # We normalize each to [0,1] using the max over the dataset, then take max.
    max_spilled = max(spilled_scores) if spilled_scores else 1.0
    max_lookahead = max(lookahead_scores) if lookahead_scores else 1.0
    combined_scores = [
        max(s / max(max_spilled, 1e-9), la / max(max_lookahead, 1e-9))
        for s, la in zip(spilled_scores, lookahead_scores)
    ]

    # --- AUROC ---
    auroc_spilled = compute_auroc(labels, spilled_scores)
    auroc_lookahead = compute_auroc(labels, lookahead_scores)
    auroc_combined = compute_auroc(labels, combined_scores)

    # --- Precision/Recall at default thresholds ---
    prec_spilled, rec_spilled = compute_precision_recall(
        labels, spilled_scores, DEFAULT_SPILLED_THRESHOLD
    )
    prec_lookahead, rec_lookahead = compute_precision_recall(
        labels, lookahead_scores, DEFAULT_LOOKAHEAD_THRESHOLD
    )

    # --- Threshold sweep for lookahead ---
    threshold_sweep = []
    for thr in THRESHOLDS:
        p, r = compute_precision_recall(labels, lookahead_scores, thr)
        threshold_sweep.append({"threshold": thr, "precision": p, "recall": r})

    # --- Summary statistics ---
    correct_lookahead = [s for la, s in zip(labels, lookahead_scores) if la == 0]
    wrong_lookahead = [s for la, s in zip(labels, lookahead_scores) if la == 1]
    mean_correct_la = sum(correct_lookahead) / len(correct_lookahead) if correct_lookahead else 0.0
    mean_wrong_la = sum(wrong_lookahead) / len(wrong_lookahead) if wrong_lookahead else 0.0

    # Targets.
    target_lookahead = 0.60
    target_combined_improvement = auroc_combined >= auroc_spilled
    target_met = auroc_lookahead >= target_lookahead and target_combined_improvement

    results_dict = {
        "experiment": "Exp 169: Lookahead Energy Pre-Filter",
        "n_questions": N_QUESTIONS,
        "n_correct": N_CORRECT,
        "n_wrong": N_WRONG,
        "vocab_size": VOCAB_SIZE,
        "n_tokens": N_TOKENS,
        "correct_peak_logit": CORRECT_PEAK_LOGIT,
        "wrong_noise_std": WRONG_NOISE_STD,
        "spilled_threshold": DEFAULT_SPILLED_THRESHOLD,
        "lookahead_threshold": DEFAULT_LOOKAHEAD_THRESHOLD,
        "auroc_spilled_only": auroc_spilled,
        "auroc_lookahead_only": auroc_lookahead,
        "auroc_combined": auroc_combined,
        "precision_spilled_at_threshold": prec_spilled,
        "recall_spilled_at_threshold": rec_spilled,
        "precision_lookahead_at_threshold": prec_lookahead,
        "recall_lookahead_at_threshold": rec_lookahead,
        "mean_lookahead_correct": mean_correct_la,
        "mean_lookahead_wrong": mean_wrong_la,
        "threshold_sweep_lookahead": threshold_sweep,
        "target_auroc_lookahead": target_lookahead,
        "target_combined_improvement": target_combined_improvement,
        "target_met": target_met,
        "per_sample": per_sample,
    }

    # --- Print summary ---
    print("Results:")
    print(
        f"  AUROC (spilled only):    {auroc_spilled:.3f}  "
        f"(baseline from Exp 157)"
    )
    print(
        f"  AUROC (lookahead only):  {auroc_lookahead:.3f}  "
        f"(target >{target_lookahead}, "
        f"{'✓ MET' if auroc_lookahead >= target_lookahead else '✗ NOT MET'})"
    )
    print(
        f"  AUROC (combined max):    {auroc_combined:.3f}  "
        f"(≥ spilled, "
        f"{'✓ MET' if target_combined_improvement else '✗ NOT MET'})"
    )
    print()
    print(f"  Precision @ spilled threshold {DEFAULT_SPILLED_THRESHOLD}:   {prec_spilled:.3f}")
    print(f"  Recall    @ spilled threshold {DEFAULT_SPILLED_THRESHOLD}:   {rec_spilled:.3f}")
    print(
        f"  Precision @ lookahead threshold {DEFAULT_LOOKAHEAD_THRESHOLD}: {prec_lookahead:.3f}"
    )
    print(
        f"  Recall    @ lookahead threshold {DEFAULT_LOOKAHEAD_THRESHOLD}: {rec_lookahead:.3f}"
    )
    print()
    print(f"  Mean lookahead energy (correct answers): {mean_correct_la:.4f} nats/token")
    print(f"  Mean lookahead energy (wrong answers):   {mean_wrong_la:.4f} nats/token")
    print()
    print("Threshold sweep (lookahead):")
    for row in threshold_sweep:
        print(f"  thr={row['threshold']:.1f}: precision={row['precision']:.3f} recall={row['recall']:.3f}")

    return results_dict


def save_results(results_dict: dict) -> Path:
    """Save results to results/experiment_169_results.json."""
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "experiment_169_results.json"
    with out_path.open("w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)

    if not results["target_met"]:
        if results["auroc_lookahead_only"] < results["target_auroc_lookahead"]:
            print(
                f"\nWARNING: Lookahead AUROC {results['auroc_lookahead_only']:.3f} "
                f"< target {results['target_auroc_lookahead']}"
            )
        if not results["target_combined_improvement"]:
            print(
                f"\nWARNING: Combined AUROC {results['auroc_combined']:.3f} "
                f"< spilled AUROC {results['auroc_spilled_only']:.3f}"
            )
        sys.exit(1)
    else:
        print(
            f"\nAll targets met: "
            f"lookahead AUROC={results['auroc_lookahead_only']:.3f}, "
            f"combined AUROC={results['auroc_combined']:.3f}"
        )
        sys.exit(0)
