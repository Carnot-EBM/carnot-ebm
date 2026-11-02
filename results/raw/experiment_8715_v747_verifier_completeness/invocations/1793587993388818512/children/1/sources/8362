#!/usr/bin/env python3
"""Exp 949 — SpilledEnergy Tier 0 pre-filter (training-free, CPU).

WHY THIS EXPERIMENT:
    arXiv 2602.18671 proposes a training-free hallucination detector that uses
    only the LLM's existing logits — no secondary model, no training, zero extra
    inference cost.  The key signal is "spilled energy": tokens whose log-probability
    is *higher* than what the contextual entropy would predict.  Hallucinations tend
    to produce overconfident, high-probability tokens that contradict context.

    Previous Tier 0 pre-filters (ThinkProbe, DRIFTProbe) both require either a
    secondary LLM call (~50–200 ms) or captured hidden states (requires model surgery).
    SpilledEnergy is cheaper: it runs entirely on the token log-probs that the
    primary LLM already computes during generation.  This makes it a natural
    first-gate before routing to Ising verification.

    This experiment validates the approach on a synthetic corpus of 200 responses
    (100 correct, 100 hallucinated) with mock logits that simulate the described
    overconfidence pattern.  Target: AUROC > 0.60.

SPEC: REQ-PROBE-022, SCENARIO-PROBE-022, SCENARIO-PROBE-023
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.spilled_energy_detector import SpilledEnergyDetector

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 949
TITLE = "SpilledEnergy Tier 0 — Training-Free Logit-Spill Hallucination Detector (CPU)"
DELIVERABLE = "results/experiment_949_spilled_energy_tier0.json"

N_CORRECT = 100
N_HALLUCINATED = 100
TOKENS_PER_RESPONSE = 20  # tokens per mock response
RANDOM_SEED = 42

# Correct responses: smooth, consistent log-probs near -2.0 (entropy ≈ 2.0)
# Every token is close to what the context predicts — low spill.
CORRECT_LOG_PROB_MEAN = -2.0
CORRECT_LOG_PROB_STD = 0.5
CORRECT_CONTEXT_ENTROPY = 2.0  # context is uncertain — expected log_p = -2.0

# Hallucinated responses: spiky log-probs with some near-zero values (overconfident)
# The model occasionally assigns very high probability to tokens that contradict context.
# mean=-1.5 (slightly higher than expected) and std=2.0 creates the overconfidence spikes.
HALLUCINATED_LOG_PROB_MEAN = -1.5
HALLUCINATED_LOG_PROB_STD = 2.0
HALLUCINATED_CONTEXT_ENTROPY = 2.0  # same context uncertainty, but model ignores it


def _generate_corpus(rng: np.random.Generator) -> tuple[list[dict], list[bool]]:
    """Generate synthetic corpus of mock LLM responses with log-probabilities.

    WHY SYNTHETIC:
        We need controlled ground truth to validate the spill metric without
        running a real LLM.  The log-prob distributions are chosen to match
        the paper's description: correct responses have smooth, predictable
        log-probs; hallucinations have occasional overconfident (near-zero)
        log-probs that 'spill' above the contextual expectation.

    Returns:
        (responses_with_logits, labels) where labels[i]=True means correct.
    """
    responses = []
    labels = []

    # Correct responses: log_probs tightly clustered around CORRECT_LOG_PROB_MEAN.
    # spill_t = max(0, lp - (-entropy)) = max(0, lp + 2.0)
    # With mean=-2.0, most tokens have lp + 2.0 ≈ 0 → near-zero spill.
    for _ in range(N_CORRECT):
        log_probs = rng.normal(
            CORRECT_LOG_PROB_MEAN, CORRECT_LOG_PROB_STD, TOKENS_PER_RESPONSE
        ).tolist()
        # Cap at 0.0: log-probs cannot be positive (probabilities ≤ 1).
        log_probs = [min(0.0, lp) for lp in log_probs]
        responses.append(
            {
                "log_probs": log_probs,
                "context_entropy": CORRECT_CONTEXT_ENTROPY,
            }
        )
        labels.append(True)

    # Hallucinated responses: broad distribution with some tokens near 0.
    # spill_t = max(0, lp + 2.0) — high-probability tokens (lp near 0) cause large spill.
    for _ in range(N_HALLUCINATED):
        log_probs = rng.normal(
            HALLUCINATED_LOG_PROB_MEAN, HALLUCINATED_LOG_PROB_STD, TOKENS_PER_RESPONSE
        ).tolist()
        log_probs = [min(0.0, lp) for lp in log_probs]
        responses.append(
            {
                "log_probs": log_probs,
                "context_entropy": HALLUCINATED_CONTEXT_ENTROPY,
            }
        )
        labels.append(False)

    return responses, labels


def main() -> None:
    """Run Exp 949: SpilledEnergy Tier 0 benchmark on synthetic corpus."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE)
    tmpl.setup()

    rng = np.random.default_rng(RANDOM_SEED)
    detector = SpilledEnergyDetector()

    with tmpl.phase("generate_synthetic_corpus"):
        responses, labels = _generate_corpus(rng)

    with tmpl.phase("compute_spill_and_benchmark"):
        result = detector.benchmark(responses, labels)

    # Compute per-class mean spill for diagnostics.
    correct_spills = [
        detector.compute_spill(r["log_probs"], r["context_entropy"])
        for r, lbl in zip(responses, labels)
        if lbl
    ]
    hallucinated_spills = [
        detector.compute_spill(r["log_probs"], r["context_entropy"])
        for r, lbl in zip(responses, labels)
        if not lbl
    ]

    payload = {
        "auroc": result.auroc,
        "optimal_threshold": result.optimal_threshold,
        "skip_rate": result.skip_rate,
        "fn_rate": result.fn_rate,
        "honest_verdict": result.honest_verdict,
        "n_correct": N_CORRECT,
        "n_hallucinated": N_HALLUCINATED,
        "n_total": N_CORRECT + N_HALLUCINATED,
        "correct_mean_spill": float(np.mean(correct_spills)),
        "hallucinated_mean_spill": float(np.mean(hallucinated_spills)),
        "spill_separation": float(np.mean(hallucinated_spills) - np.mean(correct_spills)),
        "tokens_per_response": TOKENS_PER_RESPONSE,
        "random_seed": RANDOM_SEED,
    }
    artifact = tmpl.build_result(payload, status="success")

    os.makedirs("results", exist_ok=True)
    with open(DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"AUROC: {result.auroc:.4f}")
    print(f"Optimal threshold: {result.optimal_threshold:.4f}")
    print(f"Skip rate: {result.skip_rate:.4f}")
    print(f"FN rate: {result.fn_rate:.4f}")
    print(f"Verdict: {result.honest_verdict}")
    print(f"Correct mean spill: {np.mean(correct_spills):.4f}")
    print(f"Hallucinated mean spill: {np.mean(hallucinated_spills):.4f}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
