"""Adversarial GSM8K benchmark harness (Exp 354).

**Researcher summary:**
    Apple researchers (arXiv 2410.05229) showed that appending one irrelevant sentence
    to otherwise identical math problems causes frontier LLMs to drop up to 65% accuracy
    (e.g. o1-preview: 92.7% -> 77.4%).  The core finding: LLMs attend to ALL context,
    so a distracting sentence derails their reasoning chain.

    Carnot's arithmetic verifier is a structural parser — it extracts explicit equation
    tokens (e.g. "24 + 6 = 30") and evaluates them with an Ising energy function.  The
    Ising energy is computed over the extracted constraint terms only; it is invariant to
    surrounding context words.  Therefore, Carnot SHOULD be immune to irrelevant-sentence
    injection.

    This module implements the harness: dataclasses, dataset construction, result
    aggregation, and artifact building.  Live inference is Exp 355.  This is the
    script-generation phase per research-program.md "Large Benchmark Experiments" rule.

**Detailed explanation for engineers:**
    Three-layer architecture matching the other benchmark modules (precision_benchmark,
    humaneval_live_benchmark):

    1. Data layer: AdversarialGSMQuestion wraps one question-pair (original + adversarial).
       build_adversarial_questions() appends one distractor per question from DISTRACTOR_SENTENCES
       using a seeded random.Random so results are deterministic and reproducible.

    2. Result layer: AdversarialBenchmarkResult aggregates accuracy numbers across the
       three experimental conditions (standard, adversarial, repaired-adversarial).
       compute_adversarial_results() takes three lists of (predicted_correct: bool) and
       computes all four accuracy metrics.

    3. Artifact layer: build_adversarial_artifact() produces the JSON artifact with
       schema, headline_result, honest_verdict, and robustness_invariant_holds.

    CI-safe simulated mode:
        When no live model is available (CARNOT_FORCE_LIVE != "1"), callers should use
        SYNTHETIC_CI_RESULTS (standard=0.80, adversarial=0.65, repaired=0.68, mode="simulated").
        The resulting artifact carries honest_verdict="blocked_simulated" so downstream
        tooling never confuses CI results with live GPU provenance.

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-014, SCENARIO-BENCH-015, SCENARIO-BENCH-016
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

__all__ = [
    "AdversarialGSMQuestion",
    "AdversarialBenchmarkResult",
    "SYNTHETIC_CI_RESULTS",
    "DISTRACTOR_SENTENCES",
    "build_adversarial_questions",
    "compute_adversarial_results",
    "build_adversarial_artifact",
    "MicroAdversarialResult",
    "build_micro_adversarial_artifact",
]


# ---------------------------------------------------------------------------
# Fixed distractor pool (20 sentences)
# ---------------------------------------------------------------------------

DISTRACTOR_SENTENCES: list[str] = [
    "Five of them were smaller than average.",
    "The weather was sunny that day.",
    "She had always enjoyed reading books in the evening.",
    "The train arrived three minutes early.",
    "Nobody noticed the extra chair in the corner.",
    "It was the fourth Tuesday of the month.",
    "He wore a blue jacket to the meeting.",
    "The library closes at nine o'clock on weekdays.",
    "They had visited the town before, years ago.",
    "The package was wrapped in brown paper.",
    "Her favorite color had always been green.",
    "The dog barked twice and then fell silent.",
    "Outside, the leaves were just beginning to turn.",
    "The receipt showed a total of forty-two items.",
    "He remembered to water the plants before leaving.",
    "The concert had been sold out for weeks.",
    "A red bicycle was leaning against the fence.",
    "The meeting was rescheduled to the following Thursday.",
    "She had grown up in a small coastal village.",
    "The thermometer read sixty-eight degrees at noon.",
]
"""Fixed pool of 20 distractor sentences (REQ-BENCH-006).

Why these sentences:
    They are syntactically complete, contextually plausible in everyday text, and
    semantically unrelated to arithmetic.  They contain occasional numbers
    ("forty-two items", "sixty-eight degrees") to test that the extractor does NOT
    confuse distractor numerals with equation operands.  The pool is fixed (not
    randomly generated) so experiments on different machines reproduce the same
    adversarial variants.
"""


# ---------------------------------------------------------------------------
# AdversarialGSMQuestion dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AdversarialGSMQuestion:
    """One (original, adversarial) question pair for the Apple adversarial GSM8K benchmark.

    **Why this dataclass exists:**
        The harness needs to carry both the clean question (for standard accuracy) and
        the distractor-appended variant (for adversarial accuracy) through the same
        evaluation loop.  Keeping them paired prevents accidental misalignment between
        the two conditions.

    Fields
    ------
    question_id : str
        Unique identifier, e.g. "gsm8k_0042" or "synth_0001".
    original_question : str
        The unmodified question text from GSM8K.
    adversarial_question : str
        The original question with one distractor sentence appended.
        Always contains original_question as a prefix substring.
    ground_truth_answer : str
        The canonical numeric answer as a string (e.g. "42").
    irrelevant_sentence : str
        The exact distractor sentence that was appended.  Stored here so
        results can be audited post-hoc to identify which distractor was used.

    Spec: REQ-BENCH-006, SCENARIO-BENCH-014
    """

    question_id: str
    original_question: str
    adversarial_question: str
    ground_truth_answer: str
    irrelevant_sentence: str


# ---------------------------------------------------------------------------
# build_adversarial_questions
# ---------------------------------------------------------------------------


def build_adversarial_questions(
    original_questions: list[dict[str, str]],
    *,
    seed: int = 42,
) -> list[AdversarialGSMQuestion]:
    """Append one distractor sentence per question to build the adversarial variant set.

    **Detailed explanation for engineers:**
        Uses Python's standard ``random.Random`` seeded with *seed* to assign one
        distractor from ``DISTRACTOR_SENTENCES`` to each input question.  The same
        (questions, seed) pair always produces the same output — reproducibility is
        critical for comparing across runs on different hardware.

        The distractor is appended with a single space: ``f"{original} {distractor}"``.
        This mirrors the Apple paper's methodology (one irrelevant sentence inserted
        into the problem text).

    Parameters
    ----------
    original_questions : list[dict[str, str]]
        Each dict must have at minimum keys ``"question"`` (str) and ``"answer"`` (str).
        An optional ``"question_id"`` key will be used if present; otherwise a
        zero-padded index is generated (e.g. ``"q_0042"``).
    seed : int
        PRNG seed for reproducible distractor assignment.  Default 42.

    Returns
    -------
    list[AdversarialGSMQuestion]
        One entry per input question, in the same order.

    Spec: REQ-BENCH-006, SCENARIO-BENCH-014
    """
    rng = random.Random(seed)
    result: list[AdversarialGSMQuestion] = []

    for idx, q in enumerate(original_questions):
        qid = q.get("question_id", f"q_{idx:04d}")
        original = q["question"]
        answer = q["answer"]
        distractor = rng.choice(DISTRACTOR_SENTENCES)
        adversarial = f"{original} {distractor}"
        result.append(
            AdversarialGSMQuestion(
                question_id=qid,
                original_question=original,
                adversarial_question=adversarial,
                ground_truth_answer=answer,
                irrelevant_sentence=distractor,
            )
        )

    return result


# ---------------------------------------------------------------------------
# AdversarialBenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AdversarialBenchmarkResult:
    """Aggregated accuracy metrics for the adversarial GSM8K three-condition benchmark.

    **Why three conditions:**
        standard_accuracy:            Carnot's accuracy on the original (clean) questions.
        adversarial_accuracy:         Carnot's accuracy on the distractor-appended questions.
        repaired_adversarial_accuracy: Carnot's accuracy after the verify-repair loop runs
                                      on the adversarial questions.

        accuracy_drop measures the direct degradation from distractors.
        repair_improvement measures how much the repair loop recovers.

        For the robustness claim to hold, accuracy_drop should be near zero
        (the extractor ignores context words) and repair_improvement >= 0
        (repair never makes things worse).

    Fields
    ------
    standard_accuracy : float
        Fraction correct on original (non-adversarial) questions.
    adversarial_accuracy : float
        Fraction correct on distractor-appended questions WITHOUT repair.
    accuracy_drop : float
        standard_accuracy - adversarial_accuracy.  Should be near 0 for
        a robust extractor.  Preserved without clamping — may be negative
        if adversarial somehow improves accuracy (unexpected but honest).
    repaired_adversarial_accuracy : float
        Fraction correct on adversarial questions AFTER verify-repair loop.
    repair_improvement : float
        repaired_adversarial_accuracy - adversarial_accuracy.  Positive
        means repair helped; 0 means neutral; negative means repair hurt.
    inference_mode : str
        "live_gpu" when run on real hardware with CARNOT_FORCE_LIVE=1;
        "simulated" when using synthetic CI answers.

    Spec: REQ-BENCH-006, SCENARIO-BENCH-015
    """

    standard_accuracy: float
    adversarial_accuracy: float
    accuracy_drop: float
    repaired_adversarial_accuracy: float
    repair_improvement: float
    inference_mode: str


# ---------------------------------------------------------------------------
# Synthetic CI results (safe defaults when no GPU is available)
# ---------------------------------------------------------------------------

SYNTHETIC_CI_RESULTS = AdversarialBenchmarkResult(
    standard_accuracy=0.80,
    adversarial_accuracy=0.65,
    repaired_adversarial_accuracy=0.68,
    accuracy_drop=0.15,
    repair_improvement=0.03,
    inference_mode="simulated",
)
"""Default CI-safe results returned when no live model is available.

Why these numbers:
    - standard_accuracy=0.80: plausible baseline for a small model on GSM8K.
    - adversarial_accuracy=0.65: simulates a 15-point drop (worse than a robust
      extractor should exhibit, but intentionally non-zero so the CI path exercises
      the full artifact-building logic including honest_verdict="blocked_simulated").
    - repaired_adversarial_accuracy=0.68: repair recovers 3 points (positive but
      small, so repair_improvement > 0 in simulated mode).

    NEVER use these numbers as a research result.  The artifact's honest_verdict
    will be "blocked_simulated" to prevent accidental misinterpretation.
"""


# ---------------------------------------------------------------------------
# compute_adversarial_results
# ---------------------------------------------------------------------------


def compute_adversarial_results(
    standard_correct: list[bool],
    adversarial_correct: list[bool],
    repaired_correct: list[bool],
    *,
    inference_mode: str = "simulated",
) -> AdversarialBenchmarkResult:
    """Compute accuracy metrics from three parallel lists of per-question correctness.

    **Detailed explanation for engineers:**
        Takes three boolean lists of equal length (one per experimental condition)
        and computes the four scalar accuracy metrics.  The lists must be aligned:
        ``standard_correct[i]``, ``adversarial_correct[i]``, and
        ``repaired_correct[i]`` must all refer to the SAME question_id at index *i*.

        No clamping: accuracy_drop and repair_improvement may be negative, which
        is an honest research finding (e.g. if repair accidentally breaks correct
        answers, repair_improvement < 0 must be visible).

    Parameters
    ----------
    standard_correct : list[bool]
        Per-question correctness on clean (original) questions.
    adversarial_correct : list[bool]
        Per-question correctness on distractor-appended questions (no repair).
    repaired_correct : list[bool]
        Per-question correctness on distractor-appended questions AFTER repair.
    inference_mode : str
        Passed through to AdversarialBenchmarkResult.

    Returns
    -------
    AdversarialBenchmarkResult

    Raises
    ------
    ValueError
        If the three lists have different lengths.

    Spec: REQ-BENCH-006, SCENARIO-BENCH-015
    """
    n = len(standard_correct)
    if len(adversarial_correct) != n or len(repaired_correct) != n:
        raise ValueError(
            "standard_correct, adversarial_correct, and repaired_correct must have "
            f"equal lengths; got {n}, {len(adversarial_correct)}, {len(repaired_correct)}"
        )

    if n == 0:
        return AdversarialBenchmarkResult(
            standard_accuracy=0.0,
            adversarial_accuracy=0.0,
            accuracy_drop=0.0,
            repaired_adversarial_accuracy=0.0,
            repair_improvement=0.0,
            inference_mode=inference_mode,
        )

    std_acc = sum(standard_correct) / n
    adv_acc = sum(adversarial_correct) / n
    rep_acc = sum(repaired_correct) / n

    return AdversarialBenchmarkResult(
        standard_accuracy=std_acc,
        adversarial_accuracy=adv_acc,
        accuracy_drop=std_acc - adv_acc,
        repaired_adversarial_accuracy=rep_acc,
        repair_improvement=rep_acc - adv_acc,
        inference_mode=inference_mode,
    )


# ---------------------------------------------------------------------------
# build_adversarial_artifact
# ---------------------------------------------------------------------------

#: Tolerance for the robustness invariant: adversarial_accuracy is considered
#: "not degraded" if it stays within this many percentage points of standard_accuracy.
_ROBUSTNESS_TOLERANCE = 0.05


def build_adversarial_artifact(result: AdversarialBenchmarkResult) -> dict[str, Any]:
    """Build the JSON artifact for the adversarial GSM8K benchmark result.

    **Detailed explanation for engineers:**
        Produces the structured dict that Exp 354 writes to
        ``results/experiment_354_adversarial_gsm8k_harness.json`` (harness phase) or
        Exp 355 writes to the live-inference result.

        honest_verdict logic:
            "blocked_simulated"      — inference_mode == "simulated"; no live provenance.
            "improvement_positive"   — live_gpu AND repair_improvement > 0.
            "degradation_positive"   — live_gpu AND repair_improvement <= 0 AND accuracy_drop > 0
                                       (adversarial condition made things worse; repair did not help).
            "neutral"                — live_gpu AND repair_improvement <= 0 AND accuracy_drop <= 0
                                       (extractor was fully robust; no improvement needed).

        robustness_invariant_holds:
            True when adversarial_accuracy >= standard_accuracy - _ROBUSTNESS_TOLERANCE (0.05).
            This is the primary research claim: Carnot's structural extractor degrades
            by at most 5 percentage points under irrelevant-sentence injection.

    Parameters
    ----------
    result : AdversarialBenchmarkResult

    Returns
    -------
    dict
        Artifact ready for JSON serialization.

    Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-016
    """
    # Determine honest_verdict
    if result.inference_mode == "simulated":
        honest_verdict = "blocked_simulated"
    elif result.repair_improvement > 0:
        honest_verdict = "improvement_positive"
    elif result.accuracy_drop > 0:
        honest_verdict = "degradation_positive"
    else:
        honest_verdict = "neutral"

    # Robustness invariant: extractor is "robust" if adversarial accuracy barely drops
    robustness_invariant_holds = (
        result.adversarial_accuracy >= result.standard_accuracy - _ROBUSTNESS_TOLERANCE
    )

    artifact: dict[str, Any] = {
        "schema": "carnot.adversarial_gsm8k.v1",
        "inference_mode": result.inference_mode,
        "standard_accuracy": result.standard_accuracy,
        "adversarial_accuracy": result.adversarial_accuracy,
        "accuracy_drop": result.accuracy_drop,
        "repaired_adversarial_accuracy": result.repaired_adversarial_accuracy,
        "repair_improvement": result.repair_improvement,
        "robustness_invariant_holds": robustness_invariant_holds,
        "honest_verdict": honest_verdict,
        "headline_result": {
            "standard_accuracy": result.standard_accuracy,
            "adversarial_accuracy": result.adversarial_accuracy,
            "accuracy_drop": result.accuracy_drop,
            "repair_improvement": result.repair_improvement,
            "robustness_invariant_holds": robustness_invariant_holds,
            "inference_mode": result.inference_mode,
        },
    }

    return artifact


# ---------------------------------------------------------------------------
# MicroAdversarialResult — per-model result for Exp 441 micro-benchmark
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MicroAdversarialResult:
    """Per-model aggregated result for the adversarial GSM8K micro-benchmark (Exp 441).

    **Why this dataclass exists:**
        Exp 441 scopes the adversarial benchmark to 50 questions × 3 conditions × 2 models
        to fit within the 45-minute ExperimentTimeoutWatchdog budget.  This dataclass
        carries the three per-condition accuracy values and the two derived delta fields
        in percentage-point form (matching the Apple paper's reporting style).

        ``adversarial_drop_pct`` and ``repair_improvement_pct`` are expressed in
        percentage points (accuracy × 100) so they can be directly compared to Apple's
        reported 65pp drop for frontier models.

        ``inference_mode`` must be 'live_gpu' for any non-blocked honest_verdict.
        Simulated results set this to 'simulated' and produce honest_verdict='blocked'.

    Fields
    ------
    model_id : str
        Human-readable model name (e.g. 'Gemma4-E4B-it').
    n_questions : int
        Number of GSM8K questions evaluated for this model.
    standard_accuracy : float
        Fraction correct on original (clean) questions.
    adversarial_accuracy : float
        Fraction correct on distractor-appended questions, no repair.
    repaired_accuracy : float
        Fraction correct on distractor-appended questions after VerifyRepairPipeline.
    adversarial_drop_pct : float
        (standard_accuracy - adversarial_accuracy) × 100.  Positive means adversarial
        condition hurt the model (matches Apple paper's direction).
    repair_improvement_pct : float
        (repaired_accuracy - adversarial_accuracy) × 100.  Positive means repair helped.
    inference_mode : str
        'live_gpu' when run on real GPU hardware; 'simulated' for CI/blocked paths.

    Spec: REQ-BENCH-011, SCENARIO-BENCH-029
    """

    model_id: str
    n_questions: int
    standard_accuracy: float
    adversarial_accuracy: float
    repaired_accuracy: float
    adversarial_drop_pct: float
    repair_improvement_pct: float
    inference_mode: str


# ---------------------------------------------------------------------------
# build_micro_adversarial_artifact
# ---------------------------------------------------------------------------


def build_micro_adversarial_artifact(
    results: list[MicroAdversarialResult],
) -> dict[str, Any]:
    """Build the JSON artifact for the adversarial GSM8K micro-benchmark (Exp 441).

    **Detailed explanation for engineers:**
        Assembles the artifact from a list of per-model MicroAdversarialResult objects.
        The artifact carries a single ``honest_verdict`` derived from all models:

        honest_verdict logic (in priority order):
            'blocked'               — results is empty OR any result has inference_mode != 'live_gpu'.
            'improvement_positive'  — all live_gpu AND at least one model has repair_improvement_pct > 0.
            'degradation_positive'  — all live_gpu AND no model improved AND at least one model
                                      has adversarial_drop_pct > 0 (adversarial condition hurt).
            'neutral'               — all live_gpu AND no model degraded AND no model improved
                                      (extractor fully robust; repair not needed).

        robustness_claim:
            True iff any model has repair_improvement_pct > 0 AND adversarial_drop_pct > 5.
            This is the Carnot headline: the verify-repair loop recovers accuracy on inputs
            that caused a >5pp drop from irrelevant-sentence injection.

    Parameters
    ----------
    results : list[MicroAdversarialResult]
        One entry per model evaluated.

    Returns
    -------
    dict
        Artifact ready for JSON serialization with schema='carnot.adversarial_micro.v1'.

    Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030
    """
    if not results or any(r.inference_mode != "live_gpu" for r in results):
        return {
            "schema": "carnot.adversarial_micro.v1",
            "honest_verdict": "blocked",
            "robustness_claim": False,
            "inference_mode": "blocked",
            "n_models": len(results),
            "per_model_results": [_micro_result_to_dict(r) for r in results],
            "headline_result": None,
        }

    # Derive top-level verdict from per-model results.
    any_improvement = any(r.repair_improvement_pct > 0 for r in results)
    any_degradation = any(r.adversarial_drop_pct > 0 for r in results)

    if any_improvement:
        honest_verdict = "improvement_positive"
    elif any_degradation:
        honest_verdict = "degradation_positive"
    else:
        honest_verdict = "neutral"

    robustness_claim = any(
        r.repair_improvement_pct > 0 and r.adversarial_drop_pct > 5 for r in results
    )

    avg_drop = sum(r.adversarial_drop_pct for r in results) / len(results)
    avg_improvement = sum(r.repair_improvement_pct for r in results) / len(results)
    best = max(results, key=lambda r: r.repair_improvement_pct)

    return {
        "schema": "carnot.adversarial_micro.v1",
        "honest_verdict": honest_verdict,
        "robustness_claim": robustness_claim,
        "inference_mode": "live_gpu",
        "n_models": len(results),
        "avg_adversarial_drop_pct": round(avg_drop, 2),
        "avg_repair_improvement_pct": round(avg_improvement, 2),
        "per_model_results": [_micro_result_to_dict(r) for r in results],
        "headline_result": _micro_result_to_dict(best),
    }


def _micro_result_to_dict(r: MicroAdversarialResult) -> dict[str, Any]:
    """Serialize a MicroAdversarialResult to a JSON-safe dict."""
    return {
        "model_id": r.model_id,
        "n_questions": r.n_questions,
        "standard_accuracy": r.standard_accuracy,
        "adversarial_accuracy": r.adversarial_accuracy,
        "repaired_accuracy": r.repaired_accuracy,
        "adversarial_drop_pct": r.adversarial_drop_pct,
        "repair_improvement_pct": r.repair_improvement_pct,
        "inference_mode": r.inference_mode,
    }
