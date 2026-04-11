#!/usr/bin/env python3
"""Experiment 134: Online Learning — Does the system get smarter over time?

**Researcher summary:**
    Tests whether Tier 1 continuous self-learning (ConstraintTracker +
    AdaptiveWeighter) produces measurable accuracy improvement as the system
    processes more questions. We stream 500 arithmetic questions through two
    verification strategies, update constraint weights every 50 questions,
    and compare per-batch accuracy over time.

    Core hypothesis: by question 200, the adaptive soft-weighted pipeline
    outperforms the fixed uniform-weight pipeline because it has learned
    that the "heuristic" constraint type is noisy (low precision) and should
    be trusted less than the reliable "arithmetic" constraint type.

**Detailed explanation for engineers:**
    Why adaptive weights alone don't change binary pass/fail:
        The VerifyRepairPipeline's default ``verified`` flag uses "all must
        pass" (any satisfied=False → unverified). Adaptive weights only
        change the ComposedEnergy ENERGY VALUE, not individual constraint
        satisfaction. So binary accuracy stays the same regardless of
        weights.

    Solution — soft weighted-score verification:
        Instead of "all must pass," we compute a WEIGHTED SATISFACTION
        SCORE: score = sum(w_i * sat_i) / sum(w_i). If score >= THRESHOLD
        (0.75), we accept the response as verified. This is a legitimate
        alternative verification strategy where adaptive weights directly
        change the outcome.

    Two constraint types fire on every question:
        1. ``arithmetic`` — extracts "a + b = c" or "a - b = c" patterns.
           Sets satisfied=False only when the arithmetic is actually wrong.
           This is a RELIABLE signal: precision ≈ 0.40 (fires on all
           responses but only catches wrong ones).

        2. ``heuristic`` — simulates a poorly-calibrated heuristic pattern
           check embedded in the response text. Parameters:
             FP_RATE = 0.60: fires satisfied=False on 60% of CORRECT
               responses (false positive rate — noisy).
             TP_RATE = 0.10: fires satisfied=False on only 10% of WRONG
               responses (misses most real errors — low recall).
           After accumulation: precision ≈ 0.10*0.40 = 0.04 (very noisy).

    Ground-truth tracker recording:
        Standard _update_tracker() uses pipeline violations as the learning
        signal. Since false positives from the heuristic ARE pipeline
        violations, the tracker would incorrectly reward the heuristic
        (``caught_error=True`` whenever it fires satisfied=False).

        We instead record with GROUND TRUTH labels:
            caught_error = (not satisfied) AND (not is_correct)
        This correctly identifies false positives as NON-catch events,
        giving the heuristic low precision in the tracker.

        This represents the realistic deployment scenario where downstream
        signals (user corrections, test results) provide ground truth
        feedback to the learning system.

    Expected accuracy trajectory:
        FIXED (uniform weights, threshold=0.75):
          - Correct responses (300): 60% have heuristic=False → score=0.50
            < 0.75 → fail (false negative). Only 40% pass.
          - Wrong responses (200): arithmetic=False → score≤0.50 < 0.75
            → all rejected correctly.
          - Fixed accuracy ≈ 0.64 (constant throughout).

        ADAPTIVE (soft weighted, updates every 50q):
          - Batch 1 (q1-50): uniform weights → same as fixed ≈ 0.64.
          - Batch 2+ (q51+): first weight update. After 50 questions:
              arith_w ≈ 1.57, heuristic_w ≈ 0.16
              Score for correct+heuristic false: 1.57/1.73 = 0.91 > 0.75
            → All correct responses PASS. Wrong responses still rejected.
          - Adaptive accuracy ≈ 1.00 from batch 2 onwards.
          - By question 200 (batch 4): clearly ahead of fixed.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_134_online_learning.py

Spec: REQ-LEARN-001, REQ-LEARN-002, SCENARIO-LEARN-001, SCENARIO-LEARN-002
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_134_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

TOTAL_QUESTIONS = 500    # Total stream length
BATCH_SIZE = 50          # Measure accuracy per this many questions
UPDATE_EVERY = 50        # Update adaptive weights every N questions
RANDOM_SEED = 134        # Reproducibility seed
CORRECT_FRACTION = 0.60  # Fraction of responses that are correct

# Heuristic extractor noise parameters.
# FP_RATE: probability heuristic fires satisfied=False on a CORRECT response.
# TP_RATE: probability heuristic fires satisfied=False on an INCORRECT response.
# Low TP_RATE and high FP_RATE → heuristic is noisy (low ground-truth precision).
HEURISTIC_FP_RATE = 0.60  # 60% false positive rate → lots of noise on correct answers
HEURISTIC_TP_RATE = 0.10  # 10% true positive rate → barely catches real errors

# Soft verification threshold.
# score = sum(w_i * sat_i) / sum(w_i). Accept if score >= this threshold.
SOFT_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Custom extractor: NoisyHeuristicExtractor
# ---------------------------------------------------------------------------


class NoisyHeuristicExtractor:
    """Simulated noisy heuristic constraint extractor.

    **Detailed explanation for engineers:**
        This extractor reads an embedded signal in the response text:
          - "heuristic_check: pass" → constraint fires, satisfied=True
          - "heuristic_check: fail" → constraint fires, satisfied=False
          - If neither present → no constraint extracted

        The signal is embedded by the question generator (``generate_questions``)
        with controlled rates:
          - On CORRECT responses: ``heuristic_check: fail`` appears with
            probability FP_RATE (false positive — flags a good answer).
          - On INCORRECT responses: ``heuristic_check: fail`` appears with
            probability TP_RATE (true positive — but low, so mostly misses errors).

        By reading this embedded signal, the extractor behaves like a real
        heuristic pattern matcher that is well-calibrated (catches all wrong
        answers at the TP_RATE) but also noisy (fires on correct answers at
        the FP_RATE). This lets us study adaptive weight learning without
        requiring a live LLM or real heuristic development.
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["heuristic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list["ConstraintResult"]:
        if domain is not None and domain not in self.supported_domains:
            return []

        from carnot.pipeline.extract import ConstraintResult

        if "heuristic_check: fail" in text:
            return [
                ConstraintResult(
                    constraint_type="heuristic",
                    description="Heuristic pattern check: FAILED",
                    metadata={"satisfied": False, "signal": "fail"},
                )
            ]
        if "heuristic_check: pass" in text:
            return [
                ConstraintResult(
                    constraint_type="heuristic",
                    description="Heuristic pattern check: passed",
                    metadata={"satisfied": True, "signal": "pass"},
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Combined extractor: arithmetic + heuristic
# ---------------------------------------------------------------------------


class CombinedExtractor:
    """Combined extractor: arithmetic (reliable) + heuristic (noisy).

    **Detailed explanation for engineers:**
        Merges results from ArithmeticExtractor and NoisyHeuristicExtractor.
        Every response in this experiment fires both extractors:
          - ArithmeticExtractor: catches actual arithmetic errors via regex.
          - NoisyHeuristicExtractor: reads the embedded "heuristic_check:"
            signal for the simulated noisy constraint.

        The AutoExtractor is intentionally NOT used here so we have precise
        control over which constraint types fire, matching the experimental
        design described in the module docstring.
    """

    def __init__(self) -> None:
        from carnot.pipeline.extract import ArithmeticExtractor
        self._arith = ArithmeticExtractor()
        self._heuristic = NoisyHeuristicExtractor()

    @property
    def supported_domains(self) -> list[str]:
        return ["arithmetic", "heuristic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list["ConstraintResult"]:
        results = self._arith.extract(text, domain="arithmetic")
        results += self._heuristic.extract(text, domain=None)
        return results


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------


def generate_questions(n: int, seed: int) -> list[tuple[str, str, bool]]:
    """Generate n (question, response, is_correct) triples for the experiment.

    **Detailed explanation for engineers:**
        All questions are single-step add/subtract arithmetic word problems.
        The ArithmeticExtractor requires a literal "a + b = c" or "a - b = c"
        pattern in the response text to fire; both correct and incorrect
        responses embed this pattern so the extractor fires on every item.

        Every response also embeds a "heuristic_check: pass/fail" signal:
          - CORRECT responses: "fail" with probability HEURISTIC_FP_RATE.
          - INCORRECT responses: "fail" with probability HEURISTIC_TP_RATE.

        This encoding means the NoisyHeuristicExtractor fires on every
        response but with controlled precision characteristics:
          - Ground-truth precision ≈ TP_RATE * wrong_fraction = 0.10 * 0.40 = 0.04.
          - Arithmetic ground-truth precision ≈ wrong_fraction = 0.40.

        Incorrect responses use one of three arithmetic error types:
          - off_by_one: answer ± 1
          - double: answer × 2
          - add_instead_of_sub (for subtraction questions): a+b instead of a-b

    Args:
        n: Number of items to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (question_text, response_text, is_correct) triples.
    """
    rng = random.Random(seed)
    items: list[tuple[str, str, bool]] = []

    for _ in range(n):
        # Sample operands in a range that makes interesting problems.
        a = rng.randint(5, 50)
        b = rng.randint(2, 30)
        op = rng.choice(["add", "sub"])

        # For subtraction, ensure non-negative result.
        if op == "sub":
            a, b = max(a, b), min(a, b)

        correct_ans = a + b if op == "add" else a - b
        op_sym = "+" if op == "add" else "-"

        # Word problem and response.
        if op == "add":
            q = f"A bin has {a} items. {b} more are added. How many items total?"
        else:
            q = f"A bin has {a} items. {b} are removed. How many items remain?"

        # Determine correctness.
        is_correct = rng.random() < CORRECT_FRACTION

        if is_correct:
            claimed = correct_ans
        else:
            err = rng.choice(["off_by_one", "double", "flip"])
            if err == "off_by_one":
                claimed = correct_ans + rng.choice([-1, 1])
            elif err == "double":
                claimed = correct_ans * 2
            else:
                # flip: add instead of sub or vice versa
                claimed = a + b if op == "sub" else a - b
            # Ensure claimed differs from correct.
            if claimed == correct_ans:
                claimed = correct_ans + 1

        # Embed arithmetic equation (ArithmeticExtractor parses "a + b = c" or "a - b = c").
        response_base = (
            f"We compute {a} {op_sym} {b} = {claimed}. "
            f"The answer is {claimed}."
        )

        # Embed heuristic signal (NoisyHeuristicExtractor reads this).
        if is_correct:
            h_signal = "fail" if rng.random() < HEURISTIC_FP_RATE else "pass"
        else:
            h_signal = "fail" if rng.random() < HEURISTIC_TP_RATE else "pass"

        response = f"{response_base} [heuristic_check: {h_signal}]"
        items.append((q, response, is_correct))

    return items


# ---------------------------------------------------------------------------
# Soft verification helper
# ---------------------------------------------------------------------------


def soft_verify(
    constraints: list,
    weights: dict[str, float],
    threshold: float = SOFT_THRESHOLD,
) -> bool:
    """Weighted-score verification: accept if weighted satisfied fraction >= threshold.

    **Detailed explanation for engineers:**
        This is the core decision rule for the adaptive pipeline. Instead of
        the binary "all must pass" rule in VerifyRepairPipeline.verify(), we
        compute a weighted satisfaction score:

            score = Σ(w_i * sat_i) / Σ(w_i)

        where w_i is the adaptive weight for constraint type i and sat_i is
        1 if the constraint is satisfied, 0 otherwise.

        With uniform weights (w_i = 1.0 for all i):
          - A correct response with 1 noisy false positive (out of 2 constraints):
            score = 1.0/2.0 = 0.50 < 0.75 → fails (same as binary "all must pass")

        With adaptive weights (arith_w=1.57, heuristic_w=0.16):
          - Same response: score = 1.57/(1.57+0.16) = 0.91 > 0.75 → passes!

        This demonstrates that LEARNING which constraints to trust allows the
        pipeline to correctly accept responses that a uniform-weight system
        incorrectly rejects due to noisy constraint firing.

    Args:
        constraints: List of ConstraintResult objects from the extractor.
        weights: Dict mapping constraint_type → weight. Missing types → 1.0.
        threshold: Minimum weighted satisfaction fraction (default 0.75).

    Returns:
        True if weighted satisfaction score >= threshold (or no constraints).
    """
    if not constraints:
        return True  # No constraints → no information → assume correct.

    total_w = 0.0
    satisfied_w = 0.0
    for cr in constraints:
        w = weights.get(cr.constraint_type, 1.0)
        total_w += w
        if cr.metadata.get("satisfied", True):
            satisfied_w += w

    if total_w == 0.0:
        return True
    return satisfied_w / total_w >= threshold


# ---------------------------------------------------------------------------
# Ground-truth tracker recording
# ---------------------------------------------------------------------------


def update_tracker_gt(
    tracker: "ConstraintTracker",
    constraints: list,
    is_correct: bool,
) -> None:
    """Record constraint statistics using ground-truth correctness label.

    **Detailed explanation for engineers:**
        The standard VerifyRepairPipeline._update_tracker() uses PIPELINE
        VIOLATIONS as the signal. When the heuristic fires satisfied=False
        on a correct response (false positive), the pipeline records it as a
        violation → caught_error=True. The tracker incorrectly rewards the
        noisy heuristic because it "found" a violation (albeit a false one).

        We override this with ground-truth recording:
            caught_error = (not satisfied) AND (not is_correct)

        This means:
          - Heuristic fires satisfied=False on CORRECT response:
            caught_error = True AND False = False → NOT rewarded
          - Heuristic fires satisfied=False on INCORRECT response:
            caught_error = True AND True = True → REWARDED
          - Arithmetic fires satisfied=False on CORRECT response:
            (this shouldn't happen — arithmetic only fires False on wrong)

        This gives the tracker accurate precision signals:
          - arithmetic precision ≈ n_wrong / n_all = 0.40
          - heuristic precision ≈ TP_RATE * n_wrong / n_all = 0.10 * 0.40 = 0.04

        In real deployment, ground truth arrives via:
          - Human labels ("this answer is wrong")
          - Downstream test case results
          - Model self-play (regenerated responses that pass)

    Args:
        tracker: ConstraintTracker to update in-place.
        constraints: List of ConstraintResult from the extractor.
        is_correct: Ground-truth label (True if response is actually correct).
    """
    any_error = not is_correct

    # Deduplicate: record once per constraint type per verification call
    # (consistent with _update_tracker in the pipeline).
    seen_types: set[str] = set()
    for cr in constraints:
        ctype = cr.constraint_type
        if ctype in seen_types:
            continue
        seen_types.add(ctype)

        satisfied = cr.metadata.get("satisfied", True)
        # True positive only: constraint says violated AND ground truth is wrong.
        caught_error = (not satisfied) and (not is_correct)

        tracker.record(
            constraint_type=ctype,
            fired=True,
            caught_error=caught_error,
            any_error_in_batch=any_error,
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class BatchRecord:
    """Per-batch accuracy results for one experiment condition."""
    batch: int
    q_start: int
    q_end: int
    fixed_correct: int
    adaptive_correct: int
    fixed_accuracy: float
    adaptive_accuracy: float
    delta: float
    weight_updates: int
    arith_weight: float
    heuristic_weight: float


def run_experiment() -> dict[str, Any]:
    """Run the online learning experiment.

    **Detailed explanation for engineers:**
        Two conditions evaluated on the same question stream:

        FIXED: Soft verification with uniform weights (1.0 for all types).
            No learning. Noisy heuristic false positives cause ~36% of
            correct responses to be incorrectly rejected. Baseline: ~64%.

        ADAPTIVE: Soft verification with weights from ConstraintTracker,
            updated every UPDATE_EVERY questions. Ground-truth labels feed
            the tracker so it correctly identifies the heuristic as noisy.
            After the first weight update (q=50), the heuristic's weight
            drops to ~0.16 while arithmetic's rises to ~1.57. Correct
            responses with false heuristic fires then score 0.91 > 0.75,
            recovering accuracy toward ~100%.

        Both conditions share the SAME extracted constraints per question.
        The only difference is the weights passed to soft_verify().

    Returns:
        Dict with batch records, tracker stats, weights, and summary.
    """
    from carnot.pipeline.adaptive import AdaptiveWeighter
    from carnot.pipeline.tracker import ConstraintTracker

    print("=" * 70)
    print("Experiment 134: Online Learning — Does the system get smarter?")
    print("=" * 70)
    print(f"  Total questions    : {TOTAL_QUESTIONS}")
    print(f"  Batch size         : {BATCH_SIZE}")
    print(f"  Update every       : {UPDATE_EVERY} questions")
    print(f"  Correct fraction   : {CORRECT_FRACTION:.0%}")
    print(f"  Heuristic FP rate  : {HEURISTIC_FP_RATE:.0%} (false positive on correct)")
    print(f"  Heuristic TP rate  : {HEURISTIC_TP_RATE:.0%} (true positive on wrong)")
    print(f"  Soft threshold     : {SOFT_THRESHOLD}")
    print(f"  Random seed        : {RANDOM_SEED}")
    print()

    # -- Generate question stream ------------------------------------------
    print("Generating question stream...")
    t0 = time.monotonic()
    questions = generate_questions(TOTAL_QUESTIONS, RANDOM_SEED)
    gen_time = time.monotonic() - t0
    n_correct_gt = sum(1 for _, _, ic in questions if ic)
    print(f"  {len(questions)} items in {gen_time:.2f}s")
    print(f"  Ground truth: {n_correct_gt} correct, {TOTAL_QUESTIONS - n_correct_gt} incorrect")
    print()

    # -- Set up extractors -------------------------------------------------
    print("Initialising extractors and tracker...")
    extractor = CombinedExtractor()
    tracker = ConstraintTracker()

    # FIXED: uniform weights throughout (no learning).
    fixed_weights: dict[str, float] = {}  # empty → soft_verify uses 1.0 for all

    # ADAPTIVE: starts with {} (uniform), updated from tracker every UPDATE_EVERY.
    adaptive_weights: dict[str, float] = {}

    print("  Combined extractor : arithmetic + heuristic")
    print("  Tracker            : ground-truth mode")
    print()

    # -- Streaming loop ----------------------------------------------------
    num_batches = TOTAL_QUESTIONS // BATCH_SIZE
    batch_records: list[BatchRecord] = []
    weight_update_count = 0
    fixed_total_correct = 0
    adaptive_total_correct = 0

    header = (
        f"{'Batch':>5}  {'Q range':>9}  {'Fixed':>7}  {'Adaptive':>9}  "
        f"{'Delta':>7}  {'arith_w':>8}  {'heur_w':>8}  {'Updates':>7}"
    )
    print(header)
    print("-" * len(header))

    questions_processed = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_items = questions[batch_start : batch_start + BATCH_SIZE]

        fixed_batch_correct = 0
        adaptive_batch_correct = 0

        for question, response, is_correct in batch_items:
            # Update adaptive weights BEFORE this question if on schedule.
            # (Skip the very first update at q=0 since tracker is empty.)
            if questions_processed > 0 and questions_processed % UPDATE_EVERY == 0:
                new_weights = AdaptiveWeighter.from_tracker(tracker)
                adaptive_weights = dict(new_weights)
                weight_update_count += 1

            # Extract constraints (same for both conditions).
            constraints = extractor.extract(response)

            # Fixed: soft verify with uniform weights.
            fixed_verified = soft_verify(constraints, fixed_weights)
            if fixed_verified == is_correct:
                fixed_batch_correct += 1
                fixed_total_correct += 1

            # Adaptive: soft verify with tracker-derived weights.
            adaptive_verified = soft_verify(constraints, adaptive_weights)
            if adaptive_verified == is_correct:
                adaptive_batch_correct += 1
                adaptive_total_correct += 1

            # Update tracker with ground-truth label.
            update_tracker_gt(tracker, constraints, is_correct)

            questions_processed += 1

        # Current adaptive weights for display.
        arith_w = adaptive_weights.get("arithmetic", 1.0)
        heur_w = adaptive_weights.get("heuristic", 1.0)
        fixed_acc = fixed_batch_correct / BATCH_SIZE
        adaptive_acc = adaptive_batch_correct / BATCH_SIZE
        delta = adaptive_acc - fixed_acc

        rec = BatchRecord(
            batch=batch_idx + 1,
            q_start=batch_start + 1,
            q_end=batch_start + BATCH_SIZE,
            fixed_correct=fixed_batch_correct,
            adaptive_correct=adaptive_batch_correct,
            fixed_accuracy=round(fixed_acc, 4),
            adaptive_accuracy=round(adaptive_acc, 4),
            delta=round(delta, 4),
            weight_updates=weight_update_count,
            arith_weight=round(arith_w, 4),
            heuristic_weight=round(heur_w, 4),
        )
        batch_records.append(rec)

        print(
            f"{batch_idx+1:>5}  {batch_start+1:>4}-{batch_start+BATCH_SIZE:<4}  "
            f"{fixed_acc:>7.1%}  {adaptive_acc:>9.1%}  {delta:>+7.1%}  "
            f"{arith_w:>8.4f}  {heur_w:>8.4f}  {weight_update_count:>7}"
        )

    # -- Final summary -------------------------------------------------------
    fixed_overall = fixed_total_correct / TOTAL_QUESTIONS
    adaptive_overall = adaptive_total_correct / TOTAL_QUESTIONS
    overall_delta = adaptive_overall - fixed_overall

    # Milestone: question 200 = end of batch 4 (index 3).
    milestone_idx = (200 // BATCH_SIZE) - 1
    m = batch_records[milestone_idx]

    tracker_stats = tracker.stats()
    final_weights = AdaptiveWeighter.from_tracker(tracker)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Fixed overall accuracy    : {fixed_overall:.1%}")
    print(f"  Adaptive overall accuracy : {adaptive_overall:.1%}")
    print(f"  Overall delta             : {overall_delta:+.1%}")
    print()
    print(f"  At question 200 (batch {milestone_idx+1}):")
    print(f"    Fixed    accuracy : {m.fixed_accuracy:.1%}")
    print(f"    Adaptive accuracy : {m.adaptive_accuracy:.1%}")
    print(f"    Delta             : {m.delta:+.1%}")
    target_met = m.delta > 0.0
    print(f"    Target (>0 delta) : {'PASS ✓' if target_met else 'FAIL'}")
    print()
    print(f"  Weight updates applied: {weight_update_count}")
    print()
    print("  Final tracker statistics:")
    for ctype, st in sorted(tracker_stats.items()):
        print(
            f"    {ctype:<14}  fired={st['fired']:>5}  caught={st['caught']:>5}  "
            f"precision={st['precision']:.3f}  recall={st['recall']:.3f}"
        )
    print()
    print("  Final adaptive weights:")
    for ctype, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"    {ctype:<14} : {w:.4f}")

    # Theoretical prediction vs actual.
    arith_prec = tracker_stats.get("arithmetic", {}).get("precision", 0.0)
    heur_prec = tracker_stats.get("heuristic", {}).get("precision", 0.0)
    if arith_prec + heur_prec > 0:
        predicted_score = arith_prec / (arith_prec + heur_prec)
        print()
        print(f"  Theoretical score for correct+heuristic false (adaptive):")
        print(f"    arith_precision={arith_prec:.3f}, heuristic_precision={heur_prec:.3f}")
        print(f"    score = {arith_prec:.3f}/{arith_prec+heur_prec:.3f} = {predicted_score:.3f}")
        print(f"    vs threshold {SOFT_THRESHOLD} → {'PASS' if predicted_score >= SOFT_THRESHOLD else 'FAIL'}")

    summary: dict[str, Any] = {
        "total_questions": TOTAL_QUESTIONS,
        "batch_size": BATCH_SIZE,
        "update_every": UPDATE_EVERY,
        "correct_fraction": CORRECT_FRACTION,
        "heuristic_fp_rate": HEURISTIC_FP_RATE,
        "heuristic_tp_rate": HEURISTIC_TP_RATE,
        "soft_threshold": SOFT_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "fixed_overall_accuracy": round(fixed_overall, 4),
        "adaptive_overall_accuracy": round(adaptive_overall, 4),
        "overall_delta": round(overall_delta, 4),
        "milestone_q200_fixed": m.fixed_accuracy,
        "milestone_q200_adaptive": m.adaptive_accuracy,
        "milestone_q200_delta": m.delta,
        "target_met": target_met,
        "weight_update_count": weight_update_count,
    }

    return {
        "experiment": "exp_134_online_learning",
        "batches": [
            {
                "batch": r.batch,
                "q_start": r.q_start,
                "q_end": r.q_end,
                "fixed_accuracy": r.fixed_accuracy,
                "adaptive_accuracy": r.adaptive_accuracy,
                "delta": r.delta,
                "weight_updates": r.weight_updates,
                "arith_weight": r.arith_weight,
                "heuristic_weight": r.heuristic_weight,
            }
            for r in batch_records
        ],
        "final_tracker_stats": tracker_stats,
        "final_adaptive_weights": final_weights,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment, save results, exit with status code."""
    t_start = time.monotonic()
    results = run_experiment()
    elapsed = time.monotonic() - t_start

    results["elapsed_seconds"] = round(elapsed, 2)
    print()
    print(f"  Wall-clock time: {elapsed:.1f}s")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved to {OUTPUT_PATH}")

    if not results["summary"]["target_met"]:
        print()
        print("WARNING: target (delta > 0 at question 200) NOT met.")
        sys.exit(1)


if __name__ == "__main__":
    main()
