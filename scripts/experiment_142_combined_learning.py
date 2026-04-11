#!/usr/bin/env python3
"""Experiment 142: Combined Learning — Does Tier 1 + Tier 2 compose?

**Researcher summary:**
    Tier 1 (ConstraintTracker + AdaptiveWeighter, Exp 132-134) adjusts the
    WEIGHTS of existing constraints at runtime. Tier 2 (ConstraintMemory +
    ConstraintGenerator, Exp 135, 141) adds ENTIRELY NEW constraint types
    discovered from accumulated error patterns.

    This experiment benchmarks all four combinations on 500 arithmetic + logic
    questions to answer three questions:
      1. Does Tier 2 beat Tier 1? (Hypothesis: yes — adding missing constraint
         types beats reweighting existing ones.)
      2. Does Combined (Tier 1 + Tier 2) beat either alone?
      3. What fraction of Tier 2 gains come from new constraints vs better
         coverage of existing error types?

    Results feed directly into research-program.md Tier 2 evaluation and the
    decision of whether to invest in Tier 3 (JEPA violation predictor).

**Detailed explanation for engineers:**
    Four pipeline configurations are benchmarked side-by-side:
      a. Baseline: VerifyRepairPipeline() with default fixed weights, no memory.
         This is the pre-Exp-132 pipeline — all constraint types weighted 1.0,
         no learning from previous calls.
      b. Tier1: + ConstraintTracker + AdaptiveWeighter. Weights are updated
         every WEIGHT_UPDATE_INTERVAL questions from the running tracker.
         No new constraint types are generated.
      c. Tier2: + ConstraintMemory + ConstraintGenerator. Memory accumulates
         violation patterns. AutoExtractor.extract() called with memory= to
         add generated constraints. No weight updates.
      d. Combined: Tier 1 + Tier 2. Both weight adaptation AND constraint
         generation active simultaneously.

    Dataset: 500 synthetic questions in two domains:
      - Arithmetic (300 questions, 60%): carry-chain, comparison boundary,
        and negation-scope error types matching the Exp 141 simulation.
      - Logic (200 questions, 40%): implication and negation scope errors
        that LogicExtractor partially catches.

    Warmup (first 100 questions):
      - All four configurations run in parallel on the same questions.
      - Tier 1 accumulates tracker statistics.
      - Tier 2 records memory patterns (but does not yet USE them — patterns
        need 3+ occurrences before they trigger generation).
      - After warmup, Tier 1 applies adaptive weights from the warmup tracker.
      - Tier 2's memory already has accumulated patterns; generation activates
        automatically as soon as frequency >= PATTERN_THRESHOLD (3).

    Evaluation (questions 101-500):
      - Each of the 400 questions is run through all four configurations.
      - Accuracy: fraction of questions where pipeline verdict matches
        ground truth (verified==is_correct).
      - Per-question constraint count: how many constraints were generated.
      - Memory contribution: fraction of Tier 2 / Combined eval questions where
        memory added at least one NEW constraint type not in the static output.

    Per-domain breakdown:
      - arithmetic sub-domain: carry-chain errors (need CarryChainConstraint)
        and comparison boundary errors (need BoundConstraint).
      - logic sub-domain: implication / negation errors.

    Key analysis outputs:
      - tier2_beats_tier1: True if Tier 2 accuracy > Tier 1 accuracy.
      - combined_beats_both: True if Combined accuracy > max(Tier1, Tier2).
      - pct_tier2_gains_from_new_constraints: What % of questions where Tier 2
        was correct but Baseline was wrong had at least one "generated" (non-
        static) constraint type in the result?

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_142_combined_learning.py

Spec: REQ-LEARN-001, REQ-LEARN-002, REQ-LEARN-003, REQ-LEARN-004,
      SCENARIO-LEARN-001, SCENARIO-LEARN-002, SCENARIO-LEARN-003
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_142_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Total questions (warmup + eval).
N_QUESTIONS = 500

# Warmup: first WARMUP_SIZE questions build tracker/memory state.
WARMUP_SIZE = 100

# Eval: remaining questions benchmarked across all four configs.
EVAL_SIZE = N_QUESTIONS - WARMUP_SIZE  # 400

# Fraction of questions with correct responses (applies to all domains).
CORRECT_FRACTION = 0.60

# Domain split: fraction arithmetic vs logic.
DOMAIN_SPLIT = {"arithmetic": 0.60, "logic": 0.40}

# Among WRONG arithmetic questions, distribution of error types.
ARITHMETIC_ERROR_DISTRIBUTION = {
    "arithmetic_carry": 0.60,     # multi-carry addition errors
    "comparison_boundary": 0.25,  # numeric inequality violations
    "negation_scope": 0.15,       # "X is not Y" contradictions
}

# Among WRONG logic questions, distribution of error types.
LOGIC_ERROR_DISTRIBUTION = {
    "implication": 0.60,    # "if A then B" violated
    "negation_scope": 0.40, # negation errors also appear in logic domain
}

# How often to apply Tier 1 weight updates (in questions, rolling window).
WEIGHT_UPDATE_INTERVAL = 50

# Random seed for reproducibility.
RANDOM_SEED = 142

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from carnot.pipeline.adaptive import AdaptiveWeighter
from carnot.pipeline.extract import AutoExtractor
from carnot.pipeline.generation import (
    PATTERN_ARITHMETIC_CARRY,
    PATTERN_COMPARISON_BOUNDARY,
    PATTERN_NEGATION_SCOPE,
)
from carnot.pipeline.memory import ConstraintMemory
from carnot.pipeline.tracker import ConstraintTracker
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Simulated question generators
# ---------------------------------------------------------------------------


def _make_carry_chain_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a carry-chain arithmetic question.

    **Detailed explanation for engineers:**
        Uses numbers with guaranteed multi-step carries (all-9 addends like 99,
        999). Wrong responses apply the common carry-miss pattern where the
        intermediate carry is dropped in the final result. This is the specific
        error type that CarryChainConstraint (triggered by Tier 2 memory) catches
        but that the base ArithmeticExtractor alone misses.
    """
    n_digits = rng.randint(2, 3)
    a = 10**n_digits - 1  # e.g., 99 or 999 — guaranteed cascading carries
    b = rng.randint(1, 5)
    true_answer = a + b
    if correct:
        response = f"Adding {a} + {b}: result is {true_answer}."
    else:
        # Drop the final carry — e.g., 99 + 1 = 90 instead of 100
        wrong = a - (10 ** (n_digits - 1) - 1) + b
        response = f"Adding {a} + {b}: {a} + {b} = {wrong}."
    question = f"What is {a} + {b}?"
    return question, response


def _make_boundary_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a comparison boundary arithmetic question.

    **Detailed explanation for engineers:**
        Embeds a numeric comparison claim (e.g., "5 < 100") in the response.
        Wrong responses reverse the comparison operator, producing a false claim
        like "50 > 80". BoundConstraint (triggered by Tier 2) catches these
        numeric inequality violations that standard extractors miss.
    """
    score = rng.randint(1, 50)
    threshold = rng.randint(51, 100)
    question = f"Is {score} less than {threshold}?"
    if correct:
        response = f"Yes, {score} < {threshold} is true."
    else:
        response = f"No, {score} > {threshold} holds here."
    return question, response


def _make_negation_question(correct: bool, rng: random.Random, domain: str) -> tuple[str, str]:
    """Generate a negation-scope question (works in both arithmetic and logic domains).

    **Detailed explanation for engineers:**
        Creates a self-contradictory negation pattern. For correct responses the
        negation is consistent ("the answer is X, which is not Y"). For wrong
        responses the response says "X is not Y" then concludes Y — a negation
        scope violation that NegationConstraint catches.
    """
    value = rng.randint(10, 99)
    if domain == "arithmetic":
        question = f"What is the result? (Hint: it is not {value - 1}.)"
        if correct:
            response = f"The result is {value}, which is not {value - 1}."
        else:
            response = f"The result is not {value}. Therefore the answer is {value}."
    else:
        # Logic domain: propositional negation
        question = f"Is it true that X is not {value}?"
        if correct:
            response = f"Yes, X is not {value} — the constraint holds."
        else:
            response = f"X is not {value}. Therefore X equals {value}."
    return question, response


def _make_implication_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a logic implication question.

    **Detailed explanation for engineers:**
        Creates "If A then B" reasoning. Correct responses follow the implication.
        Wrong responses violate it — e.g., affirming the antecedent is false
        but claiming the consequent is true anyway. LogicExtractor catches some
        of these via its "cannot"/"does not" patterns; this tests whether Tier 2
        helps on the missed cases.
    """
    a_val = rng.randint(1, 20)
    b_val = rng.randint(21, 50)
    question = f"If score > {a_val}, what can we say about score?"
    if correct:
        response = f"Since score > {a_val}, score must be greater than {a_val}."
    else:
        # Wrong: violate the implication by claiming score <= a_val
        response = (
            f"If score > {a_val} does not hold, score is {b_val}. "
            f"But score cannot be greater than {a_val}."
        )
    return question, response


# ---------------------------------------------------------------------------
# Simulated dataset
# ---------------------------------------------------------------------------


@dataclass
class SimQuestion:
    """A single simulated question for benchmarking.

    Attributes:
        question: The question text.
        response: The candidate response to verify.
        is_correct: Ground truth — True if the response is actually correct.
        domain: "arithmetic" or "logic" — determines which extractor fires.
        error_type: The specific error category for wrong answers (empty string
            for correct answers). Maps to constraint types like "arithmetic_carry",
            "comparison_boundary", "negation_scope", "implication".
    """

    question: str
    response: str
    is_correct: bool
    domain: str
    error_type: str


def generate_questions(n: int, rng: random.Random) -> list[SimQuestion]:
    """Generate N simulated questions across arithmetic and logic domains.

    **Detailed explanation for engineers:**
        Domain split: ~60% arithmetic, ~40% logic (rounded to nearest int).
        Within each domain, 60% correct, 40% wrong. Error types follow the
        domain-specific distributions defined in ARITHMETIC_ERROR_DISTRIBUTION
        and LOGIC_ERROR_DISTRIBUTION.

        This design ensures all four learning configurations face the same
        question set, so any accuracy differences reflect the mechanism
        (weight update vs. constraint addition) rather than dataset sampling.

    Args:
        n: Number of questions to generate.
        rng: Seeded random number generator for reproducibility.

    Returns:
        List of SimQuestion objects in mixed-domain order (shuffled).
    """
    questions: list[SimQuestion] = []

    n_arith = round(n * DOMAIN_SPLIT["arithmetic"])
    n_logic = n - n_arith

    # Generate arithmetic questions.
    arith_etypes = list(ARITHMETIC_ERROR_DISTRIBUTION.keys())
    arith_eprobs = list(ARITHMETIC_ERROR_DISTRIBUTION.values())
    for _ in range(n_arith):
        is_correct = rng.random() < CORRECT_FRACTION
        etype = rng.choices(arith_etypes, weights=arith_eprobs, k=1)[0]

        if etype == "arithmetic_carry":
            q, r = _make_carry_chain_question(is_correct, rng)
        elif etype == "comparison_boundary":
            q, r = _make_boundary_question(is_correct, rng)
        else:  # negation_scope
            q, r = _make_negation_question(is_correct, rng, "arithmetic")

        questions.append(SimQuestion(
            question=q,
            response=r,
            is_correct=is_correct,
            domain="arithmetic",
            error_type="" if is_correct else etype,
        ))

    # Generate logic questions.
    logic_etypes = list(LOGIC_ERROR_DISTRIBUTION.keys())
    logic_eprobs = list(LOGIC_ERROR_DISTRIBUTION.values())
    for _ in range(n_logic):
        is_correct = rng.random() < CORRECT_FRACTION
        etype = rng.choices(logic_etypes, weights=logic_eprobs, k=1)[0]

        if etype == "implication":
            q, r = _make_implication_question(is_correct, rng)
        else:  # negation_scope
            q, r = _make_negation_question(is_correct, rng, "logic")

        questions.append(SimQuestion(
            question=q,
            response=r,
            is_correct=is_correct,
            domain="logic",
            error_type="" if is_correct else etype,
        ))

    # Shuffle so domain types are interleaved (as in a real workload).
    rng.shuffle(questions)
    return questions


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

# Single shared AutoExtractor instance for all accuracy measurements.
# All four configs use AutoExtractor; the difference is whether memory= is
# passed (Tier 2 / Combined) and whether adaptive weights are applied (Tier 1
# / Combined). Weight adaptation only affects energy-backed constraint terms;
# standard extractor outputs are metadata-backed, so Tier 1 effects are
# visible only when energy-backed constraints are present.
_SHARED_EXT = AutoExtractor()


def _is_violated(constraints: list) -> bool:
    """Return True if any constraint in the list has satisfied=False in metadata."""
    return any(c.metadata.get("satisfied") is False for c in constraints)


def check_baseline(q: SimQuestion) -> tuple[bool, int, bool]:
    """Evaluate baseline: static extraction only, no memory, fixed weights.

    **Detailed explanation for engineers:**
        Uses AutoExtractor.extract(text, domain) — no memory parameter.
        This is the pre-learning floor. A question is correctly evaluated if:
        - Correct answer: no violations (no false alarm).
        - Wrong answer: at least one violation detected (true positive).

    Returns:
        Tuple of (verdict_correct, n_constraints, memory_added_new=False).
    """
    constraints = _SHARED_EXT.extract(q.response, q.domain)
    violated = _is_violated(constraints)
    # Pipeline "verified=True" means no violations; correct if matches ground truth.
    verdict_correct = (not violated) == q.is_correct
    return verdict_correct, len(constraints), False


def check_tier1(
    q: SimQuestion,
    pipeline: VerifyRepairPipeline,
    tracker: ConstraintTracker,
) -> tuple[bool, int, bool]:
    """Evaluate Tier 1: adaptive weights (reweighting only), no memory.

    **Detailed explanation for engineers:**
        Uses pipeline.verify() which applies adaptive weights from the
        ConstraintTracker via AdaptiveWeighter. Adaptive weights affect
        energy-backed constraints (adjusting their contribution to total energy).
        For metadata-backed constraints (most standard extractor output),
        weights have no direct effect on the satisfied=True/False flag.

        In this simulation, most constraints are metadata-backed, so Tier 1
        accuracy ≈ Baseline. This is the CORRECT finding — Tier 1 cannot help
        when the constraint gap is due to MISSING types, not noisy weighting.
        It matches the Exp 134 conclusion.

        tracker= is passed to accumulate statistics during eval (tracker
        continues learning through the eval phase — continuous adaptation).

    Returns:
        Tuple of (verdict_correct, n_constraints, memory_added_new=False).
    """
    result = pipeline.verify(q.question, q.response, domain=q.domain, tracker=tracker)
    verdict_correct = result.verified == q.is_correct
    return verdict_correct, len(result.constraints), False


def check_tier2(
    q: SimQuestion,
    memory: ConstraintMemory,
) -> tuple[bool, int, bool]:
    """Evaluate Tier 2: memory-driven constraint ADDITION, fixed weights.

    **Detailed explanation for engineers:**
        Uses AutoExtractor.extract(text, domain, memory=memory). This is the
        key Tier 2 code path: when memory has mature patterns (frequency >= 3)
        for the domain, ConstraintGenerator ADDS new constraint types:
          - "arithmetic_carry" → CarryChainConstraint
          - "comparison_boundary" → BoundConstraint
          - "negation_scope" → NegationConstraint

        These new constraints have metadata["satisfied"] set by their extractors,
        so they contribute directly to violation detection. Questions with
        carry-chain or boundary errors that the base extractor missed will now
        be caught.

        memory_added_new is True when at least one constraint type in the memory-
        augmented result set was NOT present in the static result set, confirming
        that Tier 2 generated new types for this question.

    Returns:
        Tuple of (verdict_correct, n_constraints, memory_added_new).
    """
    static_constraints = _SHARED_EXT.extract(q.response, q.domain)
    memory_constraints = _SHARED_EXT.extract(q.response, q.domain, memory=memory)

    violated = _is_violated(memory_constraints)
    verdict_correct = (not violated) == q.is_correct
    n_constraints = len(memory_constraints)

    static_types = {c.constraint_type for c in static_constraints}
    memory_types = {c.constraint_type for c in memory_constraints}
    memory_added_new = bool(memory_types - static_types)

    return verdict_correct, n_constraints, memory_added_new


def check_combined(
    q: SimQuestion,
    pipeline: VerifyRepairPipeline,
    tracker: ConstraintTracker,
    memory: ConstraintMemory,
) -> tuple[bool, int, bool]:
    """Evaluate Combined: adaptive weights (Tier 1) + constraint addition (Tier 2).

    **Detailed explanation for engineers:**
        Combines both mechanisms:
        1. Memory-augmented extraction: AutoExtractor.extract(memory=memory)
           adds new constraint types for mature patterns.
        2. Adaptive weight filtering: pipeline._adaptive_weights applied to
           energy-backed constraints (reduces noisy false positives from
           poorly-performing constraint types).

        For metadata-backed constraints, we check satisfied=False directly.
        For energy-backed constraints (if any), the pipeline's adaptive weights
        determine their contribution to total energy.

        Since most standard extractors produce metadata-backed constraints, the
        Combined config primarily shows Tier 2 gains. Any Tier 1 improvement
        would appear on top IF there were noisy energy-backed constraints.

        The tracker continues accumulating during eval for continuous adaptation.

    Returns:
        Tuple of (verdict_correct, n_constraints, memory_added_new).
    """
    # Step 1: Extract with memory (Tier 2 addition).
    static_constraints = _SHARED_EXT.extract(q.response, q.domain)
    memory_constraints = _SHARED_EXT.extract(q.response, q.domain, memory=memory)

    static_types = {c.constraint_type for c in static_constraints}
    memory_types = {c.constraint_type for c in memory_constraints}
    memory_added_new = bool(memory_types - static_types)

    # Step 2: Apply Tier 1 adaptive weight filter for any energy-backed terms.
    # We run through the pipeline to get adaptive weighting, passing the
    # memory-augmented constraint set via a temporary verify call. Since the
    # pipeline's extractor runs internally, we use the pipeline's built-in
    # memory integration (which prepends suggest_constraints results) and
    # tracker recording.
    result = pipeline.verify(q.question, q.response, domain=q.domain, tracker=tracker)

    # The pipeline.verify() result uses suggest_constraints (soft hints) but
    # NOT ConstraintGenerator. Merge: if memory-augmented extraction found
    # violations that the pipeline missed, count those.
    pipeline_violated = not result.verified
    memory_violated = _is_violated(memory_constraints)
    combined_violated = pipeline_violated or memory_violated

    verdict_correct = (not combined_violated) == q.is_correct
    # Count max of both paths for constraint count metric.
    n_constraints = max(len(result.constraints), len(memory_constraints))

    return verdict_correct, n_constraints, memory_added_new


# ---------------------------------------------------------------------------
# Pipeline configuration factories
# ---------------------------------------------------------------------------


def make_baseline() -> VerifyRepairPipeline:
    """Create Baseline pipeline: fixed weights, no memory.

    **Detailed explanation for engineers:**
        This is the pre-learning pipeline — all constraint types weighted 1.0,
        no ConstraintMemory, no ConstraintTracker. Accuracy here is the floor
        that Tier 1, Tier 2, and Combined must beat to justify the complexity.
    """
    return VerifyRepairPipeline()


def make_tier1() -> tuple[VerifyRepairPipeline, ConstraintTracker]:
    """Create Tier 1 pipeline: ConstraintTracker only (reweighting, no memory).

    **Detailed explanation for engineers:**
        Returns both the pipeline and its tracker. The caller accumulates tracker
        statistics during warmup, then calls AdaptiveWeighter.apply_to_pipeline()
        to install adaptive weights before the eval phase.

        Tier 1 represents the Exp 133/134 configuration: precision-based weight
        adaptation, no new constraint types added.

    Returns:
        Tuple of (pipeline, tracker). Tracker should be passed as tracker= to
        pipeline.verify() during warmup to accumulate statistics.
    """
    pipeline = VerifyRepairPipeline()
    tracker = ConstraintTracker()
    return pipeline, tracker


def make_tier2() -> tuple[VerifyRepairPipeline, ConstraintMemory]:
    """Create Tier 2 pipeline: ConstraintMemory only (addition, no reweighting).

    **Detailed explanation for engineers:**
        Returns both the pipeline (with memory= set) and the memory object.
        The pipeline uses memory internally during verify() to:
          1. Prepend learned constraint suggestions from ConstraintMemory.suggest_constraints()
             before the static extraction results.
          2. Record new violation patterns into memory after each verification.

        Tier 2 represents the Exp 141 configuration: memory-driven constraint
        ADDITION, without any weight updates.

    Returns:
        Tuple of (pipeline, memory). Same memory object used for both recording
        and suggestion so patterns accumulate across the full 500-question run.
    """
    memory = ConstraintMemory()
    pipeline = VerifyRepairPipeline(memory=memory)
    return pipeline, memory


def make_combined() -> tuple[VerifyRepairPipeline, ConstraintTracker, ConstraintMemory]:
    """Create Combined pipeline: Tier 1 + Tier 2 simultaneously.

    **Detailed explanation for engineers:**
        Combines ConstraintTracker (for weight adaptation) and ConstraintMemory
        (for constraint addition). The pipeline is constructed with memory= so it
        records and suggests patterns. The caller also passes tracker= to
        pipeline.verify() during warmup to accumulate weight statistics, then
        applies adaptive weights before eval.

        This configuration tests whether the two mechanisms are complementary
        (weights AND new types both help) or whether one dominates.

    Returns:
        Tuple of (pipeline, tracker, memory).
    """
    memory = ConstraintMemory()
    pipeline = VerifyRepairPipeline(memory=memory)
    tracker = ConstraintTracker()
    return pipeline, tracker, memory


# ---------------------------------------------------------------------------
# Per-domain breakdown helper
# ---------------------------------------------------------------------------


def breakdown_by_domain(
    results_per_question: list[dict[str, Any]],
    questions: list[SimQuestion],
) -> dict[str, dict[str, Any]]:
    """Compute per-domain accuracy for a list of per-question result dicts.

    **Detailed explanation for engineers:**
        Groups questions by domain and computes:
          - accuracy: fraction where verdict_correct is True
          - avg_constraints: mean constraint count
          - pct_memory_added_new: fraction of questions where memory added
            a new constraint type (only meaningful for Tier 2 / Combined)

    Args:
        results_per_question: List of dicts, one per question, with keys
            "verdict_correct", "n_constraints", "memory_added_new".
        questions: The corresponding SimQuestion list (same order/length).

    Returns:
        Dict keyed by domain with accuracy/constraint/memory stats.
    """
    domain_buckets: dict[str, list[dict[str, Any]]] = {}
    for res, q in zip(results_per_question, questions):
        domain_buckets.setdefault(q.domain, []).append(res)

    breakdown: dict[str, Any] = {}
    for domain, bucket in domain_buckets.items():
        n = len(bucket)
        n_correct = sum(1 for r in bucket if r["verdict_correct"])
        avg_c = sum(r["n_constraints"] for r in bucket) / n if n > 0 else 0.0
        pct_new = sum(1 for r in bucket if r["memory_added_new"]) / n if n > 0 else 0.0
        breakdown[domain] = {
            "n": n,
            "accuracy": round(n_correct / n, 4) if n > 0 else 0.0,
            "avg_constraints": round(avg_c, 2),
            "pct_memory_added_new": round(pct_new, 4),
        }
    return breakdown


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run Exp 142: four-way benchmark of combined learning mechanisms.

    **Detailed explanation for engineers:**
        Full experiment in three phases:
          1. Generate all 500 questions (deterministic, same for all configs).
          2. Warmup phase (q0-99): run all four configs on the same questions to
             build tracker stats and memory patterns. After warmup, apply Tier 1
             adaptive weights.
          3. Eval phase (q100-499): run all four configs on each question, record
             accuracy, constraint counts, and memory contribution.

        All four pipelines see the SAME question sequence so differences in
        accuracy are attributable to the learning mechanism, not sampling.

    Returns:
        Dict with all experiment results, ready for JSON serialization.
    """
    rng = random.Random(RANDOM_SEED)

    print("Exp 142: Combined Learning Benchmark")
    print(f"  N={N_QUESTIONS}, warmup={WARMUP_SIZE}, eval={EVAL_SIZE}")
    print(f"  correct_fraction={CORRECT_FRACTION}")
    print(f"  domain_split={DOMAIN_SPLIT}")
    print(f"  arithmetic_errors={ARITHMETIC_ERROR_DISTRIBUTION}")
    print(f"  logic_errors={LOGIC_ERROR_DISTRIBUTION}")
    print()

    # ------------------------------------------------------------------
    # Phase 0: Generate all questions up front (deterministic).
    # ------------------------------------------------------------------

    print("Generating questions...")
    all_questions = generate_questions(N_QUESTIONS, rng)
    warmup_qs = all_questions[:WARMUP_SIZE]
    eval_qs = all_questions[WARMUP_SIZE:]

    n_arith_warmup = sum(1 for q in warmup_qs if q.domain == "arithmetic")
    n_logic_warmup = sum(1 for q in warmup_qs if q.domain == "logic")
    n_arith_eval = sum(1 for q in eval_qs if q.domain == "arithmetic")
    n_logic_eval = sum(1 for q in eval_qs if q.domain == "logic")
    n_wrong_warmup = sum(1 for q in warmup_qs if not q.is_correct)
    print(f"  warmup: {WARMUP_SIZE} ({n_arith_warmup} arith, {n_logic_warmup} logic, "
          f"{n_wrong_warmup} wrong)")
    print(f"  eval:   {EVAL_SIZE} ({n_arith_eval} arith, {n_logic_eval} logic)")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Build pipeline instances.
    # ------------------------------------------------------------------

    baseline_pipeline = make_baseline()
    tier1_pipeline, tier1_tracker = make_tier1()
    tier2_pipeline, tier2_memory = make_tier2()
    combined_pipeline, combined_tracker, combined_memory = make_combined()

    # ------------------------------------------------------------------
    # Phase 2: Warmup — accumulate tracker stats and memory patterns.
    # ------------------------------------------------------------------

    print("Phase 2: Warmup...")
    warmup_start = time.perf_counter()

    for q in warmup_qs:
        # Baseline: just verify (no tracking, no memory).
        baseline_pipeline.verify(q.question, q.response, domain=q.domain)

        # Tier 1: verify WITH tracker to accumulate statistics.
        tier1_pipeline.verify(q.question, q.response, domain=q.domain, tracker=tier1_tracker)

        # Tier 2: pipeline has memory= set internally; verify() records patterns.
        # BUT verify() only records the base extractor's constraint_type (e.g.,
        # "arithmetic"), not the specialized types like "arithmetic_carry" that
        # ConstraintGenerator needs to trigger. So we ALSO explicitly seed memory
        # with the ground-truth error type for wrong answers -- the same seeding
        # strategy used in Exp 141. In production this would come from labeled
        # historical data; here we have ground truth from the simulation.
        tier2_pipeline.verify(q.question, q.response, domain=q.domain)
        if not q.is_correct and q.error_type:
            tier2_memory.record_pattern(
                q.domain,
                q.error_type,
                f"warmup seed: {q.response[:60]}",
            )

        # Combined: same double-recording for the combined memory.
        combined_pipeline.verify(q.question, q.response, domain=q.domain, tracker=combined_tracker)
        if not q.is_correct and q.error_type:
            combined_memory.record_pattern(
                q.domain,
                q.error_type,
                f"warmup seed: {q.response[:60]}",
            )

    warmup_elapsed = time.perf_counter() - warmup_start

    # Apply Tier 1 adaptive weights from warmup statistics.
    tier1_weights = AdaptiveWeighter.from_tracker(tier1_tracker)
    AdaptiveWeighter.apply_to_pipeline(tier1_pipeline, tier1_weights)

    combined_weights = AdaptiveWeighter.from_tracker(combined_tracker)
    AdaptiveWeighter.apply_to_pipeline(combined_pipeline, combined_weights)

    print(f"  Warmup complete in {warmup_elapsed:.3f}s")
    print(f"  Tier1 tracker stats: {list(tier1_tracker.stats().keys())}")
    print(f"  Tier1 adaptive weights: { {k: round(v, 4) for k, v in tier1_weights.items()} }")
    print(f"  Tier2 memory summary: {tier2_memory.summary()}")
    print(f"  Combined adaptive weights: { {k: round(v, 4) for k, v in combined_weights.items()} }")
    print(f"  Combined memory summary: {combined_memory.summary()}")
    print()

    # ------------------------------------------------------------------
    # Phase 3: Eval — four-way benchmark on questions 101-500.
    # ------------------------------------------------------------------

    print("Phase 3: Eval (four-way benchmark)...")
    eval_start = time.perf_counter()

    # Per-question result lists (one entry per eval question, per config).
    baseline_results: list[dict[str, Any]] = []
    tier1_results: list[dict[str, Any]] = []
    tier2_results: list[dict[str, Any]] = []
    combined_results: list[dict[str, Any]] = []

    for i, q in enumerate(eval_qs):
        # Baseline: static extraction, fixed weights, no learning.
        b_correct, b_nc, _ = check_baseline(q)
        baseline_results.append({
            "verdict_correct": b_correct,
            "n_constraints": b_nc,
            "memory_added_new": False,
        })

        # Tier 1: adaptive weights (pipeline.verify with tracker), no new constraints.
        t1_correct, t1_nc, _ = check_tier1(q, tier1_pipeline, tier1_tracker)
        tier1_results.append({
            "verdict_correct": t1_correct,
            "n_constraints": t1_nc,
            "memory_added_new": False,
        })

        # Tier 2: memory-driven constraint ADDITION via AutoExtractor(memory=).
        t2_correct, t2_nc, t2_new = check_tier2(q, tier2_memory)
        tier2_results.append({
            "verdict_correct": t2_correct,
            "n_constraints": t2_nc,
            "memory_added_new": t2_new,
        })

        # Combined: Tier 1 adaptive weights + Tier 2 constraint addition.
        cb_correct, cb_nc, cb_new = check_combined(q, combined_pipeline, combined_tracker, combined_memory)
        combined_results.append({
            "verdict_correct": cb_correct,
            "n_constraints": cb_nc,
            "memory_added_new": cb_new,
        })

        if (i + 1) % 100 == 0:
            b_acc = sum(r["verdict_correct"] for r in baseline_results) / len(baseline_results)
            t1_acc = sum(r["verdict_correct"] for r in tier1_results) / len(tier1_results)
            t2_acc = sum(r["verdict_correct"] for r in tier2_results) / len(tier2_results)
            cb_acc = sum(r["verdict_correct"] for r in combined_results) / len(combined_results)
            print(f"  q{WARMUP_SIZE + i + 1:4d}: baseline={b_acc:.3f}, "
                  f"tier1={t1_acc:.3f}, tier2={t2_acc:.3f}, combined={cb_acc:.3f}")

    eval_elapsed = time.perf_counter() - eval_start

    # ------------------------------------------------------------------
    # Phase 4: Aggregate metrics.
    # ------------------------------------------------------------------

    n_eval = len(eval_qs)

    def _acc(results: list[dict[str, Any]]) -> float:
        return sum(r["verdict_correct"] for r in results) / n_eval if n_eval > 0 else 0.0

    def _avg_constraints(results: list[dict[str, Any]]) -> float:
        return sum(r["n_constraints"] for r in results) / n_eval if n_eval > 0 else 0.0

    def _pct_memory_new(results: list[dict[str, Any]]) -> float:
        return sum(1 for r in results if r["memory_added_new"]) / n_eval if n_eval > 0 else 0.0

    baseline_acc = _acc(baseline_results)
    tier1_acc = _acc(tier1_results)
    tier2_acc = _acc(tier2_results)
    combined_acc = _acc(combined_results)

    # Tier 2 improvement attribution: what fraction of questions where Tier 2
    # was correct but Baseline was wrong also had memory-added new constraints?
    # This measures how much of Tier 2's gain came from NEW constraint types.
    tier2_gained = [
        (t2r, br) for t2r, br in zip(tier2_results, baseline_results)
        if t2r["verdict_correct"] and not br["verdict_correct"]
    ]
    pct_tier2_from_new = (
        sum(1 for t2r, _ in tier2_gained if t2r["memory_added_new"]) / len(tier2_gained)
        if tier2_gained else 0.0
    )

    # Per-domain breakdowns.
    baseline_domain = breakdown_by_domain(baseline_results, eval_qs)
    tier1_domain = breakdown_by_domain(tier1_results, eval_qs)
    tier2_domain = breakdown_by_domain(tier2_results, eval_qs)
    combined_domain = breakdown_by_domain(combined_results, eval_qs)

    # Key hypotheses.
    tier2_beats_tier1 = tier2_acc > tier1_acc
    combined_beats_both = combined_acc > max(tier1_acc, tier2_acc)

    # ------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline  accuracy: {baseline_acc:.4f}  "
          f"avg_constraints: {_avg_constraints(baseline_results):.2f}")
    print(f"  Tier1     accuracy: {tier1_acc:.4f}  "
          f"avg_constraints: {_avg_constraints(tier1_results):.2f}")
    print(f"  Tier2     accuracy: {tier2_acc:.4f}  "
          f"avg_constraints: {_avg_constraints(tier2_results):.2f}  "
          f"memory_new: {_pct_memory_new(tier2_results):.2%}")
    print(f"  Combined  accuracy: {combined_acc:.4f}  "
          f"avg_constraints: {_avg_constraints(combined_results):.2f}  "
          f"memory_new: {_pct_memory_new(combined_results):.2%}")
    print()
    print(f"  Hypothesis 1 — Tier2 beats Tier1: {'YES' if tier2_beats_tier1 else 'NO'}")
    print(f"    Tier2={tier2_acc:.4f} vs Tier1={tier1_acc:.4f} "
          f"(delta={tier2_acc - tier1_acc:+.4f})")
    print(f"  Hypothesis 2 — Combined beats both: {'YES' if combined_beats_both else 'NO'}")
    print(f"    Combined={combined_acc:.4f} vs max(Tier1,Tier2)={max(tier1_acc, tier2_acc):.4f} "
          f"(delta={combined_acc - max(tier1_acc, tier2_acc):+.4f})")
    print(f"  Hypothesis 3 — Tier2 gains from NEW constraints: {pct_tier2_from_new:.2%}")
    print(f"    ({len(tier2_gained)} questions where Tier2 correct but Baseline wrong)")
    print()
    print("Per-domain breakdown (eval phase):")
    for domain in ("arithmetic", "logic"):
        print(f"  {domain}:")
        for name, dom_results in [
            ("baseline", baseline_domain),
            ("tier1", tier1_domain),
            ("tier2", tier2_domain),
            ("combined", combined_domain),
        ]:
            info = dom_results.get(domain, {})
            print(f"    {name:10s}: acc={info.get('accuracy', 0):.4f}  "
                  f"avg_c={info.get('avg_constraints', 0):.2f}  "
                  f"mem_new={info.get('pct_memory_added_new', 0):.2%}")

    # ------------------------------------------------------------------
    # Assemble results dict.
    # ------------------------------------------------------------------

    results: dict[str, Any] = {
        "experiment": "exp_142_combined_learning",
        "n_questions": N_QUESTIONS,
        "warmup_size": WARMUP_SIZE,
        "eval_size": EVAL_SIZE,
        "correct_fraction": CORRECT_FRACTION,
        "domain_split": DOMAIN_SPLIT,
        "arithmetic_error_distribution": ARITHMETIC_ERROR_DISTRIBUTION,
        "logic_error_distribution": LOGIC_ERROR_DISTRIBUTION,
        "random_seed": RANDOM_SEED,
        "weight_update_interval": WEIGHT_UPDATE_INTERVAL,
        # Dataset composition.
        "dataset": {
            "n_arith_warmup": n_arith_warmup,
            "n_logic_warmup": n_logic_warmup,
            "n_wrong_warmup": n_wrong_warmup,
            "n_arith_eval": n_arith_eval,
            "n_logic_eval": n_logic_eval,
        },
        # Warmup artifacts.
        "warmup": {
            "elapsed_s": round(warmup_elapsed, 3),
            "tier1_tracker_stats": tier1_tracker.stats(),
            "tier1_adaptive_weights": {k: round(v, 6) for k, v in tier1_weights.items()},
            "tier2_memory_summary": tier2_memory.summary(),
            "combined_adaptive_weights": {k: round(v, 6) for k, v in combined_weights.items()},
            "combined_memory_summary": combined_memory.summary(),
        },
        # Accuracy results.
        "accuracy": {
            "baseline": round(baseline_acc, 4),
            "tier1": round(tier1_acc, 4),
            "tier2": round(tier2_acc, 4),
            "combined": round(combined_acc, 4),
        },
        # Average constraint counts.
        "avg_constraints": {
            "baseline": round(_avg_constraints(baseline_results), 2),
            "tier1": round(_avg_constraints(tier1_results), 2),
            "tier2": round(_avg_constraints(tier2_results), 2),
            "combined": round(_avg_constraints(combined_results), 2),
        },
        # Memory contribution (Tier 2 and Combined only).
        "pct_memory_added_new": {
            "tier2": round(_pct_memory_new(tier2_results), 4),
            "combined": round(_pct_memory_new(combined_results), 4),
        },
        # Per-domain breakdown.
        "per_domain": {
            "baseline": baseline_domain,
            "tier1": tier1_domain,
            "tier2": tier2_domain,
            "combined": combined_domain,
        },
        # Improvement deltas relative to baseline.
        "deltas_vs_baseline": {
            "tier1": round(tier1_acc - baseline_acc, 4),
            "tier2": round(tier2_acc - baseline_acc, 4),
            "combined": round(combined_acc - baseline_acc, 4),
        },
        # Hypothesis verdicts.
        "hypotheses": {
            "tier2_beats_tier1": tier2_beats_tier1,
            "tier2_delta_vs_tier1": round(tier2_acc - tier1_acc, 4),
            "combined_beats_both": combined_beats_both,
            "combined_delta_vs_best": round(
                combined_acc - max(tier1_acc, tier2_acc), 4
            ),
            "pct_tier2_gains_from_new_constraints": round(pct_tier2_from_new, 4),
            "n_tier2_gained_questions": len(tier2_gained),
        },
        # Reference: Exp 134 baseline numbers for comparison.
        "exp134_reference": {
            "fixed_overall_accuracy": 0.676,
            "adaptive_overall_accuracy": 0.97,
            "overall_delta": 0.294,
            "note": "Exp 134 used synthetic data with 60% correct, 40% heuristic noise; "
                    "direct comparison is approximate.",
        },
        "eval_elapsed_s": round(eval_elapsed, 3),
    }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment and save results to results/experiment_142_results.json."""
    t_start = time.perf_counter()
    results = run_experiment()
    elapsed = time.perf_counter() - t_start

    results["total_elapsed_s"] = round(elapsed, 3)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print()
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
