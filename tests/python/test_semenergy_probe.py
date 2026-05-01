"""Tests for SemEnergyProbe — arXiv 2508.14496 Boltzmann energy probe.

All tests reference:
  REQ-TIER0-006: SemEnergyProbe must compute E(x) = -log Z(x) from logit vectors.
  SCENARIO-TIER0-006-A: Higher per-word energy for uncertain/hallucinating outputs.
  SCENARIO-TIER0-006-B: Lower per-word energy for confident/correct outputs.
  SCENARIO-TIER0-006-C: Inference time < 5 ms per example in proxy mode.
  SCENARIO-TIER0-006-D: AUROC > 0.70 on FoVer corpus in proxy mode.
  SCENARIO-TIER0-006-E: compute_energy formula matches arXiv 2508.14496 paper.

These tests cover only the new code added in this experiment (semenergy_probe.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FOVER_PATH = Path(__file__).parents[2] / "data" / "fover_corpus_v4.json"


def _load_fover() -> list[dict]:
    """Load FoVer corpus; skip if file missing (CI without large data)."""
    if not _FOVER_PATH.exists():
        pytest.skip("FoVer corpus not found — skipping corpus-level test.")
    with _FOVER_PATH.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def probe():
    """Default SemEnergyProbe instance (T=1.0, top_k=50)."""
    from carnot.verify.semenergy_probe import SemEnergyProbe

    return SemEnergyProbe(temperature=1.0, top_k=50)


# ---------------------------------------------------------------------------
# SCENARIO-TIER0-006-A: Higher energy for uncertain outputs
# ---------------------------------------------------------------------------


def test_semenergy_higher_for_uncertain_outputs(probe):
    """Math-dense correct CoT step must have lower (more negative) energy than
    a verbose prose step with few unique numbers.

    REQ-TIER0-006, SCENARIO-TIER0-006-A.

    The probe assigns high-logit proxy entries to each *unique* numeric value.
    A step that introduces many distinct numbers (correct math derivation) has
    higher unique-number density per word → larger Z per word → more negative
    per-word energy (more confident).  A step that uses few unique values
    in many words (verbose, repetitive) gets higher per-word energy (uncertain).
    """
    # Correct-style: short, dense math — introduces 5 distinct numbers, 5 operators
    math_dense = (
        "Step 1: 6 × 7 = 42.  "
        "Step 2: 42 + 18 = 60.  "
        "Step 3: 60 / 3 = 20.  "
        "Therefore the answer is 20."
    )
    # Incorrect-style: long, repetitive — reuses the same two numbers many times
    # with many prose words but few unique mathematical commitments per word.
    math_sparse = (
        "To find the answer, we need to consider that the value of x is 5 and "
        "the value of x remains 5 throughout all the steps.  The value x is "
        "still 5.  We use x = 5.  Given that x equals 5, and x is 5, "
        "the final answer based on x being 5 is that x = 5."
    )
    e_dense = probe.score_response_proxy(math_dense)
    e_sparse = probe.score_response_proxy(math_sparse)
    assert e_dense < e_sparse, (
        f"Expected dense-math energy ({e_dense:.3f}) < sparse-math energy "
        f"({e_sparse:.3f}), but dense was higher.  "
        "Check that unique-number counting lowers per-word energy."
    )


# ---------------------------------------------------------------------------
# SCENARIO-TIER0-006-B: Lower energy for confident outputs
# ---------------------------------------------------------------------------


def test_semenergy_lower_for_confident_outputs(probe):
    """A tight multi-step derivation with many unique numbers must rank as more
    confident (lower energy) than a hedged, vague description.

    REQ-TIER0-006, SCENARIO-TIER0-006-B.

    This tests the core discriminative direction: confident numeric reasoning
    should receive negative per-word energy well below the uncertain baseline.
    """
    # Confident: introduces 7 unique values with explicit arithmetic
    confident = (
        "The total cost is 3 × $42 = $126.  "
        "Adding $49.50 + $67.50 = $117.  "
        "Grand total: $117 + $126 = $243."
    )
    # Uncertain: 2 unique values buried in verbose hedged prose (many words, few nums)
    uncertain = (
        "There are a number of factors to consider here.  "
        "Perhaps the value might be approximately somewhere around 5 or possibly 6, "
        "depending on various assumptions and conditions that may or may not apply "
        "to the particular context of this problem.  It is unclear how to proceed "
        "given the ambiguity in the problem statement."
    )
    e_confident = probe.score_response_proxy(confident)
    e_uncertain = probe.score_response_proxy(uncertain)
    assert e_confident < e_uncertain, (
        f"Expected confident energy ({e_confident:.3f}) < uncertain energy "
        f"({e_uncertain:.3f}).  The confident step introduces 7 unique values "
        "and should have lower per-word energy."
    )


# ---------------------------------------------------------------------------
# SCENARIO-TIER0-006-C: Inference time < 5 ms per example
# ---------------------------------------------------------------------------


def test_semenergy_inference_time_under_5ms(probe):
    """Single proxy score must complete in < 5 ms (median over 20 calls).

    REQ-TIER0-006, SCENARIO-TIER0-006-C.

    The probe is designed to run as a fast Tier 0 gate (no GPU, no model
    loading).  5 ms is a generous budget for pure-Python text scoring.
    """
    text = (
        "To find the total, multiply 6 × $55 = $330, "
        "then add $204 + $160 + $330 = $694.  "
        "Therefore the total cost is $694."
    )
    # Warm up to avoid first-call import overhead.
    probe.score_response_proxy(text)

    times_ms = []
    for _ in range(20):
        _, elapsed_ms = probe.timed_score_proxy(text)
        times_ms.append(elapsed_ms)

    median_ms = float(np.median(times_ms))
    assert median_ms < 5.0, f"Median proxy inference time {median_ms:.2f} ms exceeds 5 ms budget."


# ---------------------------------------------------------------------------
# SCENARIO-TIER0-006-D: AUROC > 0.70 on FoVer corpus (proxy mode)
# ---------------------------------------------------------------------------


def test_semenergy_auroc_above_07_on_fover(probe):
    """Proxy-mode AUROC must exceed 0.70 on a stratified FoVer sample.

    REQ-TIER0-006, SCENARIO-TIER0-006-D.

    Uses a stratified sample (all 114 incorrect + 386 correct = 500 total)
    for stable AUROC estimation.  The per-word Boltzmann proxy achieves
    ~0.95 AUROC empirically; the test asserts the conservative 0.70 floor.
    """
    from sklearn.metrics import roc_auc_score

    corpus = _load_fover()
    incorrect = [x for x in corpus if x["label"] == "incorrect"]
    correct = [x for x in corpus if x["label"] == "correct"]

    n_correct_needed = 500 - len(incorrect)
    rng = np.random.default_rng(42)
    correct_sample_idx = rng.choice(
        len(correct), size=min(n_correct_needed, len(correct)), replace=False
    )
    eval_rows = incorrect + [correct[i] for i in correct_sample_idx]

    scores = [probe.score_response_proxy(row["step_text"]) for row in eval_rows]
    # Label: 1 = hallucinating (incorrect), 0 = correct.
    # Higher energy (less negative per-word E) → more hallucinating.
    labels = [1 if row["label"] == "incorrect" else 0 for row in eval_rows]

    auroc = roc_auc_score(labels, scores)
    assert auroc >= 0.70, (
        f"Proxy-mode AUROC {auroc:.4f} is below the 0.70 target.  "
        "Unique-number density signal may have regressed — check "
        "_NUMBER_PATTERN and unique-number counting logic."
    )


# ---------------------------------------------------------------------------
# SCENARIO-TIER0-006-E: compute_energy formula matches the paper
# ---------------------------------------------------------------------------


def test_compute_energy_formula_matches_manual(probe):
    """E = -log Σ_k exp(l_k / T) matches manual calculation.

    REQ-TIER0-006, SCENARIO-TIER0-006-E.

    Validates that compute_energy() implements the paper formula exactly,
    not an approximation, using a hand-computed reference with T=1.0.
    """
    # Logit array: [10, 8, 5, 3, 1] with top_k=5
    logits = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
    energy = probe.compute_energy(logits)
    z_manual = sum(math.exp(l) for l in [10.0, 8.0, 5.0, 3.0, 1.0])
    expected = -math.log(z_manual)
    assert abs(energy - expected) < 1e-6, f"compute_energy {energy:.6f} != manual {expected:.6f}"
