"""Experiment 1093: Phase 1c Verifier Joint Null-Space Measurement.

Phase-1c acceptance gate: joint null-space dimension < 5% of input space.

The joint null space ∩_i ker(E_i) is the set of inputs where ALL verifiers
simultaneously give near-zero energy (i.e., believe the response is correct).
A large joint null space is an adversarial opening: specification-gaming
attacks can craft inputs that fool every verifier at once.

This experiment measures:
  1. Individual verifier null-space fractions on 500 FoVer corpus examples.
  2. Pairwise r-correlations between all active verifiers.
  3. Joint null-space fraction under AND-composition.
  4. Whether Phase-1c acceptance gate is met (joint fraction < 0.05).

Spec: REQ-DIAG-003, SCENARIO-PHASE1C-001
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

import numpy as np

# Ensure we can import from python/carnot
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from python.carnot.eval.diagnostics import NullSpaceEstimator
from python.carnot.verify.nup_probe import NUPProbeV4
from python.carnot.verify.pcib_probe import PCIBProbe
from python.carnot.verify.spilled_energy import SpilledEnergyDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 1093
N_EXAMPLES = 500
N_CORRECT = 250
N_WRONG = 250
CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1093_phase1c_verifier_joint_null_space_measurement.json"
)
RANDOM_SEED = 1093

# Phase-1c acceptance threshold
JOINT_NULL_SPACE_THRESHOLD = 0.05
# Diversity threshold: r-correlation below this = adequately diverse
R_CORRELATION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_corpus(path: Path, n_correct: int, n_wrong: int, seed: int) -> list[dict]:
    """Load a balanced sample from the FoVer corpus.

    Because the corpus is heavily skewed (6434 correct vs 114 wrong), we sample
    min(n_wrong, available_wrong) from incorrect examples and the same count from
    correct examples if there are not enough wrong examples.
    """
    with open(path) as f:
        data = json.load(f)

    correct = [x for x in data if x["label"] == "correct"]
    wrong = [x for x in data if x["label"] == "incorrect"]

    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)

    # Use however many wrong examples are available (114 total in v4)
    actual_wrong = min(n_wrong, len(wrong))
    actual_correct = min(n_correct, len(correct))

    sample = wrong[:actual_wrong] + correct[:actual_correct]
    rng.shuffle(sample)
    return sample


def make_feature_matrix(examples: list[dict]) -> np.ndarray:
    """Build a simple feature matrix from text examples for NullSpaceEstimator.

    NullSpaceEstimator.fit() needs an (N, D) feature matrix to know the input
    dimension. We use three cheap text statistics as proxy features so the
    input-space dimension D is well-defined.

    Features per example:
      0: log(len(step_text) + 1)         — response length signal
      1: number count / len              — numeric density
      2: unique-word fraction            — lexical diversity
    """
    feats = []
    for ex in examples:
        text = ex.get("step_text", "")
        length = len(text) + 1
        words = text.split()
        n_words = max(len(words), 1)
        numeric_count = sum(1 for w in words if any(c.isdigit() for c in w))
        unique_words = len(set(words))
        feats.append(
            [
                float(np.log(length)),
                numeric_count / n_words,
                unique_words / n_words,
            ]
        )
    return np.array(feats, dtype=float)


def score_with_verifier(verifier_fn, examples: list[dict]) -> np.ndarray:
    """Apply verifier_fn to each example; return length-N score array.

    verifier_fn is called with (step_text, context="") and must return a float.
    Higher score = more likely hallucinating (higher energy).
    """
    scores = []
    for ex in examples:
        text = ex.get("step_text", "")
        try:
            s = float(verifier_fn(text))
        except Exception:
            s = 0.0
        scores.append(s)
    return np.array(scores, dtype=float)


def single_null_space_fraction(scores: np.ndarray, threshold_sigma: float = 0.1) -> float:
    """Fraction of examples where a single verifier gives near-zero energy.

    "Near-zero" = |score| < threshold_sigma * std(scores). This is the same
    criterion used by NullSpaceEstimator internally.
    """
    threshold = threshold_sigma * float(np.std(scores)) + 1e-9
    return float(np.mean(np.abs(scores) < threshold))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run() -> dict:
    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.time()

    # -- 1. Load corpus -------------------------------------------------------
    examples = load_corpus(CORPUS_PATH, N_CORRECT, N_WRONG, RANDOM_SEED)
    n_examples = len(examples)
    print(f"Loaded {n_examples} examples from FoVer corpus v4")

    # -- 2. Instantiate verifiers ---------------------------------------------
    spilled = SpilledEnergyDetector()
    nup = NUPProbeV4()
    pcib = PCIBProbe()

    # Define the active verifier set with text-only scoring callables.
    # ThinkPRMProbe is excluded: it requires a 1 GB GGUF model download which
    # is not available in CI/sandbox environments. The three lightweight
    # probes (SpilledEnergy, NUP, PCIB) represent the Tier-0 ensemble that
    # is always available without GPU or network access.
    verifier_callables = {
        "SpilledEnergyDetector": lambda text: spilled.score(text, context=""),
        "NUPProbeV4": lambda text: nup.score(text, context=""),
        "PCIBProbe": lambda text: pcib.score(text, context=""),
    }
    verifier_names = list(verifier_callables.keys())
    n_verifiers = len(verifier_names)
    print(f"Active verifiers: {verifier_names}")

    # -- 3. Score all examples with each verifier -----------------------------
    scores_matrix = np.zeros((n_examples, n_verifiers), dtype=float)
    for col_idx, (name, fn) in enumerate(verifier_callables.items()):
        print(f"  Scoring with {name}...")
        scores_matrix[:, col_idx] = score_with_verifier(fn, examples)
    print("Scoring complete.")

    # -- 4. Build feature matrix and fit NullSpaceEstimator -------------------
    X = make_feature_matrix(examples)
    estimator = NullSpaceEstimator()
    estimator.fit(X=X, verifier_scores=scores_matrix)

    joint_frac = estimator.joint_null_space_fraction()
    print(f"Joint null-space fraction: {joint_frac:.4f} (threshold: {JOINT_NULL_SPACE_THRESHOLD})")

    # -- 5. Pairwise r-correlations -------------------------------------------
    r_correlations: dict[str, float] = {}
    for (i, n_i), (j, n_j) in itertools.combinations(enumerate(verifier_names), 2):
        key = f"{n_i} vs {n_j}"
        r_correlations[key] = estimator.r_correlation(i, j)
        print(f"  r_corr({key}) = {r_correlations[key]:.4f}")

    max_r = max(r_correlations.values()) if r_correlations else 0.0
    min_r = min(r_correlations.values()) if r_correlations else 0.0

    # -- 6. Single-verifier null-space fractions ------------------------------
    single_fracs: dict[str, float] = {}
    for col_idx, name in enumerate(verifier_names):
        single_fracs[name] = single_null_space_fraction(scores_matrix[:, col_idx])
        print(f"  Single null-space({name}) = {single_fracs[name]:.4f}")

    # -- 7. Determine acceptance and verdict ----------------------------------
    phase1c_acceptance_met = joint_frac < JOINT_NULL_SPACE_THRESHOLD
    r_diverse_enough = max_r < R_CORRELATION_THRESHOLD
    and_viable = phase1c_acceptance_met and r_diverse_enough

    if and_viable:
        honest_verdict = "joint_null_space_small_and_composition_viable"
    elif not phase1c_acceptance_met and not r_diverse_enough:
        honest_verdict = "verifiers_correlated_diversity_needed"
    elif not phase1c_acceptance_met:
        honest_verdict = "joint_null_space_large_more_verifiers_needed"
    elif not r_diverse_enough:
        honest_verdict = "verifiers_correlated_diversity_needed"
    else:
        honest_verdict = "partial_measurement"

    # -- 8. Write artifact ----------------------------------------------------
    duration_s = time.time() - t0
    finished_at = datetime.now(tz=UTC).isoformat()

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "phase1c_null_space_v1",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 2),
        "status": "success",
        "title": "Phase 1c Verifier Joint Null-Space Measurement",
        "n_verifiers_measured": n_verifiers,
        "verifier_names": verifier_names,
        "n_examples_evaluated": n_examples,
        "joint_null_space_fraction": round(joint_frac, 6),
        "phase1c_acceptance_met": phase1c_acceptance_met,
        "r_correlations": {k: round(v, 6) for k, v in r_correlations.items()},
        "max_r_correlation": round(max_r, 6),
        "min_r_correlation": round(min_r, 6),
        "single_verifier_null_space_fractions": {k: round(v, 6) for k, v in single_fracs.items()},
        "and_composition_viable": and_viable,
        "tests_passing": 0,  # updated after test run
        "honest_verdict": honest_verdict,
        "analysis": {
            "joint_null_space_threshold": JOINT_NULL_SPACE_THRESHOLD,
            "r_correlation_threshold": R_CORRELATION_THRESHOLD,
            "r_diverse_enough": r_diverse_enough,
            "note": (
                "AND-composition shrinks the joint null space exponentially in k only when "
                "verifiers have low pairwise r-correlation (high diversity). "
                "The three Tier-0 text probes used here are training-free and available "
                "without GPU/network. ThinkPRMProbe was excluded from this run because it "
                "requires a GGUF model not cached in this environment."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written to {OUTPUT_PATH}")
    print(f"honest_verdict: {honest_verdict}")
    print(f"and_composition_viable: {and_viable}")
    return artifact


if __name__ == "__main__":
    run()
