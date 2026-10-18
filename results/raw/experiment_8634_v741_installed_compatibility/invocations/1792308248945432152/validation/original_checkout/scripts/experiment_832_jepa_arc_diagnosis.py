#!/usr/bin/env python3
"""Exp 832 — JEPA v23 ARC-Challenge Collapse Diagnosis.

**Researcher summary:**
    Exp 825 showed JEPA v23 scored auc_arc=0.04 (far below random 0.50) while
    auc_humaneval=0.76.  Root-cause hypothesis: the LIMO training corpus (Exp 824)
    contained ZERO ARC-Challenge planning examples, so ARC-style text falls outside
    the TF-IDF vocabulary and produces near-zero embeddings — making cosine distance
    uninformative (random or anti-correlated).

    This experiment:
    1. Loads the JEPA v23 pickle from results/jepa_v23_limo_model.pkl.
    2. Generates 10 synthetic correct + 10 synthetic incorrect test steps per domain
       (GSM8K arithmetic, HumanEval code, ARC-Challenge planning).
    3. For each step: computes JEPA energy, 8-dim feature norm proxy.
    4. Builds per-domain diagnosis_finding with mean_score_correct,
       mean_score_incorrect, variance, feature_norm, is_anti_correlated,
       is_uncertain, n_arc_training_pairs.
    5. Emits honest_verdict: arc_diagnosis_found / arc_diagnosis_uncertain /
       arc_unexpected_viable.

**Why this tells us the root cause:**
    If ARC feature vectors are near-zero (TF-IDF produces zeros for OOV tokens),
    then cosine distance(anchor, step) is undefined / 0.0 for ALL steps regardless
    of correctness.  The model cannot discriminate correct from incorrect ARC steps
    at all — both score ~0.0.  Depending on internal normalisation this can produce:
    - Uncertain: mean_correct ≈ mean_incorrect (model just guesses randomly)
    - Anti-correlated: mean_correct < mean_incorrect (model inverted — wrong direction)
    Both produce AUC << 0.5.

Spec: REQ-LEARN-832-001, SCENARIO-LEARN-832-001
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

# Ensure project root on PYTHONPATH for imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

# Each domain has 10 "correct" step texts and 10 "incorrect" step texts.
# These are surface-representative examples — they test whether the JEPA v23
# TF-IDF vocabulary covers these token types or not.

SYNTHETIC_STEPS: dict[str, dict[str, list[str]]] = {
    "gsm8k": {
        "correct": [
            "Step 1: 47 + 28 = 75, so the running total is 75.",
            "Multiply both sides by 3: 3 * 12 = 36.",
            "Divide the total by 4: 60 / 4 = 15.",
            "Subtract 18 from 45: 45 - 18 = 27.",
            "The cost is $5 per item and we buy 7 items: 5 * 7 = 35.",
            "Add the partial sums: 100 + 35 + 7 = 142.",
            "There are 24 hours in a day, so 3 days = 72 hours.",
            "15% of 200 is 0.15 * 200 = 30.",
            "Round 3.7 to the nearest integer: result is 4.",
            "The remainder of 17 / 5 is 2.",
        ],
        "incorrect": [
            "Step 1: 47 + 28 = 65, so the running total is 65.",
            "Multiply both sides by 3: 3 * 12 = 40.",
            "Divide the total by 4: 60 / 4 = 20.",
            "Subtract 18 from 45: 45 - 18 = 30.",
            "The cost is $5 per item and we buy 7 items: 5 * 7 = 42.",
            "Add the partial sums: 100 + 35 + 7 = 150.",
            "There are 24 hours in a day, so 3 days = 60 hours.",
            "15% of 200 is 0.15 * 200 = 40.",
            "Round 3.7 to the nearest integer: result is 3.",
            "The remainder of 17 / 5 is 3.",
        ],
    },
    "humaneval": {
        "correct": [
            "def add(a, b): return a + b",
            "def is_even(n): return n % 2 == 0",
            "result = sorted(items, key=lambda x: x[1])",
            "for i in range(len(arr)): arr[i] *= 2",
            "return [x for x in lst if x > 0]",
            "if len(s) == 0: return True",
            "total = sum(values)",
            "d = {k: v for k, v in zip(keys, vals)}",
            "class Node: def __init__(self, val): self.val = val",
            "assert output == expected, f'got {output}'",
        ],
        "incorrect": [
            "def add(a, b): return a - b",
            "def is_even(n): return n % 2 == 1",
            "result = sorted(items, key=lambda x: x[0])",
            "for i in range(len(arr)): arr[i] += 2",
            "return [x for x in lst if x < 0]",
            "if len(s) == 0: return False",
            "total = max(values)",
            "d = {v: k for k, v in zip(keys, vals)}",
            "class Node: def __init__(self, val): self.value = val",
            "assert output != expected, f'got {output}'",
        ],
    },
    "arc": {
        "correct": [
            "Since A implies B and B implies C, we can conclude that A implies C by transitivity.",
            "All mammals are warm-blooded; a whale is a mammal; therefore a whale is warm-blooded.",
            "The hypothesis contradicts the premise, so the argument is invalid.",
            "By elimination: if not P and not Q then the only remaining option is R.",
            "If the evidence supports both X and Y, we choose the simpler explanation (Occam).",
            "The conclusion follows from the two premises via modus ponens.",
            "Because heat rises, the upper layers will be warmer — consistent with observation.",
            "Photosynthesis requires sunlight; the plant is in darkness; therefore it cannot photosynthesize.",
            "The pattern increases by doubling: 1, 2, 4, 8, so the next term is 16.",
            "Since all birds have feathers and robins are birds, robins must have feathers.",
        ],
        "incorrect": [
            "Since A implies B and B implies C, we cannot conclude anything about A and C.",
            "All mammals are warm-blooded; a whale is a mammal; therefore a whale is cold-blooded.",
            "The hypothesis supports the premise, so the argument is valid.",
            "By elimination: if not P and not Q then the only remaining option is P.",
            "If the evidence supports both X and Y, we must choose the more complex explanation.",
            "The conclusion contradicts the two premises despite modus ponens.",
            "Because heat rises, the lower layers will be warmer — inconsistent with observation.",
            "Photosynthesis requires sunlight; the plant is in darkness; therefore it can photosynthesize.",
            "The pattern increases by doubling: 1, 2, 4, 8, so the next term is 10.",
            "Since all birds have feathers and robins are birds, robins need not have feathers.",
        ],
    },
}

# Prefix texts used as the "anchor" when calling predict_energy(prefix, step).
# These represent the question/context the step is supposed to answer.
DOMAIN_PREFIXES: dict[str, str] = {
    "gsm8k": "Solve this arithmetic word problem step by step.",
    "humaneval": "Write a Python function that satisfies the following docstring.",
    "arc": "Answer this science / logic question by reasoning step by step.",
}

# How many ARC training pairs were in the Exp 824 corpus (from the JSON).
N_ARC_TRAINING_PAIRS = 0  # confirmed from experiment_824_jepa_v23_limo_corpus.json


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _compute_feature_vector(model: Any, prefix: str, step: str) -> list[float]:
    """Compute an 8-dimensional feature vector for a (prefix, step) pair.

    **Why 8 dimensions:**
        We want enough dimensions to characterise the embedding geometry without
        requiring the full 64-D space.  The 8 features capture:
        - Embedding magnitudes (step and prefix separately)
        - Vocabulary coverage (fraction of tokens in vocab)
        - Cosine distance (the raw JEPA score)
        - Derived statistics (variance, range of step embedding)

    Args:
        model: JEPAv23Predictor with _vectoriser and encode() method.
        prefix: Question/context string (anchor).
        step:   Step text to characterise.

    Returns:
        8-float list: [step_norm, prefix_norm, vocab_coverage,
                       cosine_dist, step_embed_var, step_embed_range,
                       dot_product, l2_dist]
    """
    step_vec = model._vectoriser.transform(step)
    prefix_vec = model._vectoriser.transform(prefix)

    # Embedding norms (TF-IDF space before the linear layer).
    step_norm = math.sqrt(sum(v * v for v in step_vec))
    prefix_norm = math.sqrt(sum(v * v for v in prefix_vec))

    # Fraction of step tokens that appear in the trained TF-IDF vocabulary.
    tokens = model._vectoriser._tokenise(step)
    vocab = model._vectoriser._vocab
    coverage = sum(1 for t in tokens if t in vocab) / max(len(tokens), 1) if tokens else 0.0

    # Raw JEPA energy (cosine distance between prefix and step embeddings).
    cosine_dist = model.predict_energy(prefix, step)

    # Step embedding (post linear + ReLU, 64-D).
    step_embed = model.encode(step)
    step_embed_var = _variance(step_embed)
    step_embed_range = max(step_embed) - min(step_embed)

    # Dot product between prefix and step embeddings (before normalisation).
    prefix_embed = model.encode(prefix)
    dot = sum(a * b for a, b in zip(step_embed, prefix_embed))

    # L2 distance in embedding space.
    l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(step_embed, prefix_embed)))

    return [
        step_norm,
        prefix_norm,
        coverage,
        cosine_dist,
        step_embed_var,
        step_embed_range,
        dot,
        l2,
    ]


def _variance(xs: list[float]) -> float:
    """Population variance of a list of floats."""
    if not xs:
        return 0.0
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def _mean(xs: list[float]) -> float:
    """Arithmetic mean of a list; returns 0.0 for empty."""
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Per-domain diagnosis
# ---------------------------------------------------------------------------


def diagnose_domain(
    model: Any,
    domain: str,
    correct_steps: list[str],
    incorrect_steps: list[str],
    prefix: str,
) -> dict[str, Any]:
    """Run JEPA scoring on correct vs incorrect steps for one domain.

    **What this tells us:**
        - mean_score_correct: average cosine distance for correct steps.
          Lower distance = model thinks the step is more aligned with the prefix.
        - mean_score_incorrect: average cosine distance for incorrect steps.
          For a working model: mean_score_incorrect > mean_score_correct.
        - is_anti_correlated: True if mean_correct > mean_incorrect + 0.02
          (model assigns HIGHER distance to correct steps — completely inverted).
        - is_uncertain: True if |mean_correct - mean_incorrect| < 0.02
          (model cannot discriminate at all).
        - feature_norm: mean TF-IDF norm of step vectors — near-zero means OOV.

    Args:
        model:            Trained JEPAv23Predictor.
        domain:           Domain label ("gsm8k", "humaneval", "arc").
        correct_steps:    10 correct step texts.
        incorrect_steps:  10 incorrect step texts.
        prefix:           Anchor text for predict_energy().

    Returns:
        Dict with diagnosis fields.
    """
    correct_scores = [model.predict_energy(prefix, s) for s in correct_steps]
    incorrect_scores = [model.predict_energy(prefix, s) for s in incorrect_steps]

    correct_features = [_compute_feature_vector(model, prefix, s) for s in correct_steps]
    incorrect_features = [_compute_feature_vector(model, prefix, s) for s in incorrect_steps]

    # Feature norm = mean TF-IDF norm (index 0 in feature vector).
    all_norms = [f[0] for f in correct_features] + [f[0] for f in incorrect_features]
    mean_feature_norm = _mean(all_norms)

    # Vocab coverage = mean fraction of tokens in vocabulary.
    all_coverage = [f[2] for f in correct_features] + [f[2] for f in incorrect_features]
    mean_vocab_coverage = _mean(all_coverage)

    mean_correct = _mean(correct_scores)
    mean_incorrect = _mean(incorrect_scores)
    all_scores = correct_scores + incorrect_scores
    score_variance = _variance(all_scores)

    # Anti-correlated: correct steps score HIGHER (worse) than incorrect steps.
    is_anti_correlated = mean_correct > mean_incorrect + 0.02
    # Uncertain: model cannot discriminate (scores are nearly equal).
    is_uncertain = abs(mean_correct - mean_incorrect) < 0.02

    return {
        "mean_score_correct": round(mean_correct, 6),
        "mean_score_incorrect": round(mean_incorrect, 6),
        "score_delta": round(mean_incorrect - mean_correct, 6),
        "variance": round(score_variance, 6),
        "mean_feature_norm": round(mean_feature_norm, 6),
        "mean_vocab_coverage": round(mean_vocab_coverage, 6),
        "is_anti_correlated": is_anti_correlated,
        "is_uncertain": is_uncertain,
        "is_working": (not is_anti_correlated and not is_uncertain),
        "n_arc_training_pairs": N_ARC_TRAINING_PAIRS if domain == "arc" else None,
        "correct_scores": [round(s, 6) for s in correct_scores],
        "incorrect_scores": [round(s, 6) for s in incorrect_scores],
    }


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------


def build_recommendation(findings: dict[str, Any]) -> str:
    """Translate diagnosis findings into a concrete fix recommendation.

    **Logic:**
        - If ARC is_anti_correlated: add 50+ ARC training pairs to flip the signal.
        - If ARC feature_norm near-zero (< 0.01): add 20+ ARC pairs to seed the vocab.
        - If ARC is_uncertain: add 30+ ARC pairs — borderline case.
        - If ARC is_working (unexpected): the collapse must have another cause.

    Args:
        findings: Dict mapping domain → diagnosis dict.

    Returns:
        Human-readable recommendation string.
    """
    arc = findings.get("arc", {})
    if not arc:
        return "Could not diagnose ARC — insufficient data."

    norm = arc.get("mean_feature_norm", 1.0)
    anti = arc.get("is_anti_correlated", False)
    uncertain = arc.get("is_uncertain", False)

    if norm < 0.01:
        return (
            "ARC feature vectors are near-zero (mean_feature_norm < 0.01) — "
            "ARC planning vocabulary is completely out-of-distribution for the LIMO corpus TF-IDF. "
            "Add >= 50 ARC training pairs in Exp 834 to seed the vocabulary and re-train."
        )
    if anti:
        return (
            "ARC is anti-correlated (mean_score_correct > mean_score_incorrect): "
            "the model inverts correct/incorrect for planning steps. "
            "Add >= 50 ARC training pairs in Exp 834 to correct training distribution."
        )
    if uncertain:
        return (
            "ARC scores are near-random (|mean_correct - mean_incorrect| < 0.02): "
            "model has no signal on planning steps. "
            "Add >= 30 ARC training pairs in Exp 834 to introduce planning signal."
        )
    return (
        "ARC scores appear reasonable (unexpected — collapse may have a different cause). "
        "Investigate ARC evaluation logic in Exp 825 before adding training data."
    )


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------


def compute_honest_verdict(findings: dict[str, Any]) -> str:
    """Map diagnosis findings to one of three honest verdict labels.

    **Labels:**
        - arc_diagnosis_found: we identified the collapse root cause (anti-correlated
          or near-zero features).
        - arc_diagnosis_uncertain: diagnosis is inconclusive.
        - arc_unexpected_viable: ARC scores are actually reasonable (AUC collapse
          was not caused by this mechanism).

    Args:
        findings: Dict mapping domain → diagnosis dict.

    Returns:
        One of the three verdict strings.
    """
    arc = findings.get("arc", {})
    if not arc:
        return "arc_diagnosis_uncertain"

    anti = arc.get("is_anti_correlated", False)
    norm = arc.get("mean_feature_norm", 1.0)
    uncertain = arc.get("is_uncertain", False)
    working = arc.get("is_working", True)

    if anti or norm < 0.01:
        return "arc_diagnosis_found"
    if uncertain:
        return "arc_diagnosis_found"  # also a clear failure mode
    if working:
        return "arc_unexpected_viable"
    return "arc_diagnosis_uncertain"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ARC diagnosis experiment end-to-end."""
    tmpl = ExperimentTemplate(
        exp_id=832,
        title="JEPA v23 ARC-Challenge Collapse Diagnosis",
        deliverable="results/experiment_832_jepa_arc_collapse_diagnosis.json",
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(832, timeout_minutes=30)
    watchdog.start()

    try:
        _run(tmpl)
    finally:
        watchdog.stop()


def _run(tmpl: ExperimentTemplate) -> None:
    """Inner implementation, separated so tests can call without watchdog."""
    # ------------------------------------------------------------------
    # Phase 1: Load JEPA v23 model
    # ------------------------------------------------------------------
    with tmpl.phase("load_model"):
        model_path = Path("results/jepa_v23_limo_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(
                f"JEPA v23 checkpoint not found at {model_path}. "
                "Run Exp 824 first to produce this file."
            )
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)  # noqa: S301 — trusted project artifact

        # Confirm it has the expected interface.
        if not (hasattr(model, "predict_energy") and hasattr(model, "encode")):
            raise TypeError(
                f"Loaded object of type {type(model)} does not expose "
                "predict_energy() and encode() — is this the right file?"
            )

    # ------------------------------------------------------------------
    # Phase 2: Per-domain diagnosis
    # ------------------------------------------------------------------
    with tmpl.phase("diagnose_domains"):
        diagnosis_finding: dict[str, Any] = {}

        for domain, steps in SYNTHETIC_STEPS.items():
            prefix = DOMAIN_PREFIXES[domain]
            finding = diagnose_domain(
                model=model,
                domain=domain,
                correct_steps=steps["correct"],
                incorrect_steps=steps["incorrect"],
                prefix=prefix,
            )
            diagnosis_finding[domain] = finding

    # ------------------------------------------------------------------
    # Phase 3: Build recommendation and verdict
    # ------------------------------------------------------------------
    with tmpl.phase("build_verdict"):
        recommendation = build_recommendation(diagnosis_finding)
        honest_verdict = compute_honest_verdict(diagnosis_finding)

    # ------------------------------------------------------------------
    # Phase 4: Write artifact
    # ------------------------------------------------------------------
    with tmpl.phase("write_artifact"):
        result = tmpl.build_result(
            {
                "diagnosis_finding": diagnosis_finding,
                "recommendation": recommendation,
                "honest_verdict": honest_verdict,
                "n_domains": len(diagnosis_finding),
                "n_steps_per_domain": 20,  # 10 correct + 10 incorrect
                "model_path": "results/jepa_v23_limo_model.pkl",
                "exp_824_n_arc_training_pairs": N_ARC_TRAINING_PAIRS,
                "exp_824_auc_arc": 0.04,
                "exp_824_auc_humaneval": 0.76,
                "exp_824_auc_gsm8k": 0.36,
            },
            status="success",
            decision_class="detect",
        )

        out_path = Path("results/experiment_832_jepa_arc_collapse_diagnosis.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
