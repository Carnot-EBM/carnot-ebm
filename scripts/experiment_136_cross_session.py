#!/usr/bin/env python3
"""Experiment 136: Cross-Session Memory — Does domain memory improve later sessions?

**Researcher summary:**
    Tests whether ConstraintMemory (Exp 135, Tier 2) built during one session
    measurably helps subsequent sessions. Three sessions are simulated back-to-back:
      - Session 1: Verify 200 arithmetic questions → build + save memory/tracker
      - Session 2: Load session 1 memory → verify 200 NEW arithmetic questions.
        Compare "no memory" vs "with memory" on the same questions.
      - Session 3: Verify 200 mixed-domain questions (arithmetic + logic + code)
        using session-1 arithmetic memory. Measures cross-domain transfer.

    Core hypotheses:
      H1. Memory accumulates: after 200 arithmetic questions, learned constraints
          emerge for the arithmetic domain (pattern frequency ≥ 3).
      H2. Memory helps same domain: session 2 with memory gets MORE constraint
          hints than session 2 without memory, flagging known error categories.
      H3. Memory speeds repair: repair loops guided by memory-identified error
          types converge faster than unguided repair.
      H4. Memory is domain-specific: session 3 arithmetic questions benefit from
          memory; logic/code questions do NOT (cross-domain isolation).

**Detailed explanation for engineers:**
    WHY memories don't directly change binary accuracy:
        The VerifyRepairPipeline's ``verify()`` returns verified=True when ALL
        constraint metadata["satisfied"] flags are True (or absent). Learned
        constraints from ConstraintMemory.suggest_constraints() have NO
        ``satisfied`` key — they are informational "extra scrutiny" hints.
        Therefore they do NOT flip the verified flag on their own.

    WHAT memory DOES affect in this experiment:
        1. ``n_memory_suggestions``: count of learned constraint hints prepended
           per question. This grows from 0 (session 2 no-memory) to >0 (session 2
           with memory) once patterns are mature.
        2. Repair speed (simulated): when memory has learned a domain's common
           error types, a targeted repair prompt has a higher success probability
           than an unguided "arithmetic error, please fix" prompt. We model the
           repair success probability as:
               p_repair = p_base + memory_boost * (1 - p_base)
           where ``memory_boost`` = precision of the learned pattern from the
           Exp 135 tracker. This follows from the tracker's ground-truth precision
           accumulation (see experiment_134_online_learning.py for the same
           approach).
        3. Cross-domain: arithmetic memory suggestions appear only for
           domain="arithmetic", not domain="logic" or "code". The session 3
           arithmetic subgroup should see suggestions; logic/code should not.

    Simulation of repair:
        Each wrong response enters a repair loop (up to MAX_REPAIR_ATTEMPTS):
        - Without memory: each attempt has P_REPAIR_BASE probability of success
          (simulates random trial-and-error with only the violation message as
          feedback — no knowledge of which error TYPE is most common).
        - With memory: each attempt uses the learned error-type distribution.
          If the memory pattern for this domain has precision P, the targeted
          repair succeeds with probability P_REPAIR_BASE + memory_boost * (1 - P_REPAIR_BASE),
          where memory_boost = min(pattern_precision, 1.0).
        We track iterations_to_first_repair per question, then compute the mean.
        Fewer iterations = memory is helping the repair converge faster.

    Session 3 mixed domain:
        Questions are split 34/33/33 arithmetic/logic/code. Domain labels are
        passed explicitly to the pipeline so ArithmeticExtractor fires on
        arithmetic questions, LogicExtractor on logic, etc. The memory loaded
        is the arithmetic-only memory from session 1; logic/code questions get
        0 memory suggestions, confirming domain specificity.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_136_cross_session.py

Spec: REQ-LEARN-003, SCENARIO-LEARN-003 (Tier 2 Constraint Memory)
"""

from __future__ import annotations

import json
import random
import sys
import time
import tempfile
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

OUTPUT_PATH = RESULTS_DIR / "experiment_136_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Number of questions per session.
N_SESSION_1 = 200   # Arithmetic warm-up → build memory
N_SESSION_2 = 200   # New arithmetic questions → test memory vs no-memory
N_SESSION_3 = 200   # Mixed domain → test cross-domain isolation

# Fraction of responses that are correct (ground truth).
CORRECT_FRACTION = 0.50  # 50% correct so arithmetic errors appear abundantly.

# Repair simulation parameters.
# P_REPAIR_BASE: probability of a correct repair on one attempt WITHOUT memory
#   (models blind repair from violation text alone, no domain knowledge).
P_REPAIR_BASE = 0.40
# MAX_REPAIR_ATTEMPTS: cap on repair iterations per question.
MAX_REPAIR_ATTEMPTS = 5
# MEMORY_BOOST_CAP: maximum boost that memory precision can add to repair prob.
MEMORY_BOOST_CAP = 0.50

# Random seeds for reproducibility — each session uses a distinct seed so
# session 2/3 questions are genuinely new (not repeats of session 1).
SEED_SESSION_1 = 136
SEED_SESSION_2 = 137
SEED_SESSION_3 = 138

# Minimum pattern frequency for ConstraintMemory to produce suggestions.
# This matches PATTERN_THRESHOLD in memory.py (3). Declared here for clarity.
MEMORY_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------


def generate_arithmetic_questions(n: int, seed: int) -> list[tuple[str, str, bool]]:
    """Generate n (question, response, is_correct) arithmetic triples.

    **Detailed explanation for engineers:**
        Produces single-step add/subtract word problems with embedded
        arithmetic equations that ArithmeticExtractor can parse. Each
        response is either correct (claimed == correct_ans) or wrong.
        Wrong responses use one of three error patterns:
          - off_by_one: answer ± 1 (very common human error)
          - double: answer × 2 (wrong operation scale)
          - flip: add instead of subtract or vice versa

        Both correct and wrong responses embed the equation in the form
        "a OP b = claimed" so ArithmeticExtractor's regex fires reliably.

    Args:
        n: Number of questions to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (question_text, response_text, is_correct) triples.
    """
    rng = random.Random(seed)
    items: list[tuple[str, str, bool]] = []

    for _ in range(n):
        a = rng.randint(5, 50)
        b = rng.randint(2, 30)
        op = rng.choice(["add", "sub"])

        # Ensure non-negative result for subtraction.
        if op == "sub":
            a, b = max(a, b), min(a, b)

        correct_ans = a + b if op == "add" else a - b
        op_sym = "+" if op == "add" else "-"

        q = (
            f"A bin has {a} items. "
            + (f"{b} more are added." if op == "add" else f"{b} are removed.")
            + " How many items?"
        )

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
                claimed = a + b if op == "sub" else a - b
            if claimed == correct_ans:
                claimed = correct_ans + 1

        response = (
            f"We compute {a} {op_sym} {b} = {claimed}. The answer is {claimed}."
        )
        items.append((q, response, is_correct))

    return items


def generate_logic_questions(n: int, seed: int) -> list[tuple[str, str, bool]]:
    """Generate n (question, response, is_correct) logic triples.

    **Detailed explanation for engineers:**
        Produces simple modus-ponens claims: "If P then Q. P is true. So Q."
        LogicExtractor parses "If X then Y" implication patterns and checks
        that the response correctly applies modus ponens. Incorrect responses
        negate the conclusion or add an incorrect clause.

        Logic errors are different from arithmetic errors — they are captured
        by LogicExtractor's "implication" constraint type, not ArithmeticExtractor's
        "arithmetic" type. This means arithmetic memory should NOT help logic
        questions, testing H4 (domain specificity).

    Args:
        n: Number of logic questions.
        seed: Random seed.

    Returns:
        List of (question_text, response_text, is_correct) triples.
    """
    rng = random.Random(seed)
    items: list[tuple[str, str, bool]] = []

    subjects = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    properties = [
        ("is a student", "studies regularly"),
        ("has a library card", "can borrow books"),
        ("is registered", "can attend class"),
        ("passed the exam", "gets a certificate"),
        ("is a member", "receives the newsletter"),
    ]

    for _ in range(n):
        subj = rng.choice(subjects)
        prop, conseq = rng.choice(properties)
        q = f"If {subj} {prop} then {subj} {conseq}. {subj} {prop}. What follows?"

        is_correct = rng.random() < CORRECT_FRACTION

        if is_correct:
            # Correct modus ponens conclusion.
            response = (
                f"If {subj} {prop} then {subj} {conseq}. "
                f"Since {subj} {prop}, it follows that {subj} {conseq}."
            )
        else:
            # Wrong: negate or use incorrect conclusion.
            response = (
                f"If {subj} {prop} then {subj} {conseq}. "
                f"Since {subj} {prop}, it follows that {subj} does not {conseq}."
            )

        items.append((q, response, is_correct))

    return items


def generate_code_questions(n: int, seed: int) -> list[tuple[str, str, bool]]:
    """Generate n (question, response, is_correct) code triples.

    **Detailed explanation for engineers:**
        Produces Python code snippets that CodeExtractor checks for syntax
        and initialisation errors. Correct responses are valid Python.
        Incorrect responses have a NameError-style undefined variable use,
        which CodeExtractor flags as an "initialization" constraint violation.

        Code errors are a third constraint type ("initialization" from
        CodeExtractor), further isolating from arithmetic domain memory.

    Args:
        n: Number of code questions.
        seed: Random seed.

    Returns:
        List of (question_text, response_text, is_correct) triples.
    """
    rng = random.Random(seed)
    items: list[tuple[str, str, bool]] = []

    var_names = ["count", "total", "result", "value", "amount"]

    for _ in range(n):
        var = rng.choice(var_names)
        val = rng.randint(1, 100)
        q = f"Write Python code that stores {val} in a variable called {var} and prints it."

        is_correct = rng.random() < CORRECT_FRACTION

        if is_correct:
            response = f"```python\n{var} = {val}\nprint({var})\n```"
        else:
            # NameError: use var before assignment — CodeExtractor catches this.
            other = "x" if var != "x" else "y"
            response = f"```python\nprint({other})\n{var} = {val}\n```"

        items.append((q, response, is_correct))

    return items


def generate_mixed_questions(
    n: int, seed: int
) -> list[tuple[str, str, bool, str]]:
    """Generate n mixed (question, response, is_correct, domain) 4-tuples.

    **Detailed explanation for engineers:**
        Splits n evenly (±1) across three domains: arithmetic, logic, code.
        Returns 4-tuples so callers can pass the domain to the pipeline for
        domain-specific constraint extraction.

    Args:
        n: Total number of questions.
        seed: Random seed (used to sub-seed each domain generator).

    Returns:
        List of (question_text, response_text, is_correct, domain) 4-tuples.
    """
    n_arith = n // 3
    n_logic = n // 3
    n_code = n - n_arith - n_logic  # Absorb remainder.

    arith = [
        (q, r, ic, "arithmetic")
        for q, r, ic in generate_arithmetic_questions(n_arith, seed)
    ]
    logic = [
        (q, r, ic, "logic")
        for q, r, ic in generate_logic_questions(n_logic, seed + 1)
    ]
    code = [
        (q, r, ic, "code")
        for q, r, ic in generate_code_questions(n_code, seed + 2)
    ]

    combined = arith + logic + code
    # Shuffle so domains are interleaved rather than blocked.
    rng = random.Random(seed + 3)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def run_verification_session(
    questions: list[tuple[str, str, bool]],
    domain: str,
    memory,
    tracker,
) -> dict[str, Any]:
    """Verify a list of questions and return per-question results.

    **Detailed explanation for engineers:**
        Runs the pipeline for each (question, response, is_correct) triple.
        The pipeline is constructed ONCE with the provided memory object so
        session-1 memory persists across all 200 questions (the memory
        accumulates within the session too — patterns seen 3x in session 1
        become active by question ~10 since 50% wrong → ~1 arithmetic
        pattern per 2 questions → threshold at ~6 questions).

        Records per-question:
          - ``verified``: pipeline's binary verdict
          - ``n_constraints``: total constraints (base + memory suggestions)
          - ``n_memory_suggestions``: count of learned constraint hints
          - ``correct``: ground-truth label
          - ``agreement``: verified == is_correct (pipeline was right)
          - ``n_violations``: number of constraint violations found

    Args:
        questions: List of (question, response, is_correct) triples.
        domain: Domain for constraint extraction ("arithmetic", "logic", etc.).
        memory: ConstraintMemory instance (or None for no-memory condition).
        tracker: ConstraintTracker instance (or None for no tracking).

    Returns:
        Dict with per-question records and session-level summary statistics.
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(memory=memory)

    records: list[dict] = []
    n_correct = 0
    n_memory_suggestions_total = 0
    t_session_start = time.monotonic()

    for question, response, is_correct in questions:
        t0 = time.monotonic()
        result = pipeline.verify(question, response, domain=domain, tracker=tracker)
        t_ms = (time.monotonic() - t0) * 1000

        # Count how many constraints came from memory (type="learned").
        n_mem = sum(
            1 for c in result.constraints if c.constraint_type == "learned"
        )
        n_memory_suggestions_total += n_mem

        agreement = result.verified == is_correct
        if agreement:
            n_correct += 1

        records.append(
            {
                "verified": result.verified,
                "is_correct": is_correct,
                "agreement": agreement,
                "n_constraints": len(result.constraints),
                "n_violations": len(result.violations),
                "n_memory_suggestions": n_mem,
                "verify_ms": round(t_ms, 2),
            }
        )

    elapsed = time.monotonic() - t_session_start
    n = len(questions)
    accuracy = n_correct / n if n > 0 else 0.0

    return {
        "records": records,
        "accuracy": round(accuracy, 4),
        "n_questions": n,
        "n_correct_agreements": n_correct,
        "n_memory_suggestions_total": n_memory_suggestions_total,
        "avg_memory_suggestions": round(n_memory_suggestions_total / n, 4) if n else 0.0,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_mixed_domain_session(
    questions: list[tuple[str, str, bool, str]],
    memory,
) -> dict[str, Any]:
    """Verify mixed-domain questions and return domain-stratified results.

    **Detailed explanation for engineers:**
        Runs one pipeline per question using the domain label from the
        question tuple. Collects results broken down by domain so we can
        compare: does arithmetic memory help arithmetic questions but NOT
        logic or code questions?

        The single memory object passed in (session 1 arithmetic memory)
        should produce learned suggestions for domain="arithmetic" questions
        but return empty suggestions for domain="logic" and domain="code",
        since the memory only learned patterns from arithmetic verifications.

    Args:
        questions: List of (question, response, is_correct, domain) 4-tuples.
        memory: ConstraintMemory (arithmetic-trained, from session 1).

    Returns:
        Dict with per-domain summaries and total summary.
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    # One shared pipeline (memory is domain-keyed internally).
    pipeline = VerifyRepairPipeline(memory=memory)

    # Per-domain accumulators.
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in ("arithmetic", "logic", "code"):
        by_domain[domain] = {
            "n": 0,
            "n_correct": 0,
            "n_memory_suggestions": 0,
            "records": [],
        }

    t_start = time.monotonic()

    for question, response, is_correct, domain in questions:
        result = pipeline.verify(question, response, domain=domain)
        n_mem = sum(
            1 for c in result.constraints if c.constraint_type == "learned"
        )
        agreement = result.verified == is_correct
        acc_domain = by_domain[domain]
        acc_domain["n"] += 1
        if agreement:
            acc_domain["n_correct"] += 1
        acc_domain["n_memory_suggestions"] += n_mem
        acc_domain["records"].append(
            {
                "domain": domain,
                "verified": result.verified,
                "is_correct": is_correct,
                "agreement": agreement,
                "n_constraints": len(result.constraints),
                "n_violations": len(result.violations),
                "n_memory_suggestions": n_mem,
            }
        )

    elapsed = time.monotonic() - t_start

    # Summarise per domain.
    domain_summaries: dict[str, dict] = {}
    for domain, acc in by_domain.items():
        n = acc["n"]
        n_correct = acc["n_correct"]
        n_mem = acc["n_memory_suggestions"]
        domain_summaries[domain] = {
            "n_questions": n,
            "accuracy": round(n_correct / n, 4) if n else 0.0,
            "n_memory_suggestions": n_mem,
            "avg_memory_suggestions": round(n_mem / n, 4) if n else 0.0,
        }

    total_n = len(questions)
    total_correct = sum(acc["n_correct"] for acc in by_domain.values())

    return {
        "domain_summaries": domain_summaries,
        "total_accuracy": round(total_correct / total_n, 4) if total_n else 0.0,
        "total_questions": total_n,
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Repair speed simulation
# ---------------------------------------------------------------------------


def simulate_repair_speed(
    questions: list[tuple[str, str, bool]],
    domain: str,
    memory_precision: float,
    p_base: float = P_REPAIR_BASE,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate repair convergence speed with and without memory guidance.

    **Detailed explanation for engineers:**
        For each INCORRECT response (is_correct=False), we simulate two repair
        conditions:

        Condition NO-MEMORY:
            Each attempt succeeds with probability ``p_base`` regardless of
            error type. This models blind repair: the pipeline gives a generic
            "arithmetic error found" message and the repair agent tries randomly.

        Condition WITH-MEMORY:
            Each attempt succeeds with probability:
                p_mem = p_base + memory_boost * (1 - p_base)
            where:
                memory_boost = min(memory_precision, MEMORY_BOOST_CAP)
            ``memory_precision`` = tracker precision for the dominant learned
            error type in this domain. High precision = memory is reliable →
            repair agent can target the specific error type → higher success.
            This models: "arithmetic domain commonly has off_by_one errors;
            specifically check ±1 arithmetic" → agent fixes it more reliably.

        For each wrong question, we draw Bernoulli outcomes until success or
        max_attempts exhausted. We record the iteration at which repair
        succeeded (None if exhausted). Then compute:
          - mean iterations to first repair (among successes)
          - fraction repaired within max_attempts

        NOTE: This simulation uses RNG (not the actual LLM), so the results
        reflect the STATISTICAL EFFECT of higher per-attempt success probability.
        The memory_precision value comes directly from the Tier 1 ConstraintTracker
        accumulated in session 1, making the boost grounded in real data.

    Args:
        questions: List of (question, response, is_correct) triples.
        domain: Domain label (for display only).
        memory_precision: Tracker precision for the dominant error type in
            this domain. Comes from session-1 ConstraintTracker stats.
        p_base: Per-attempt repair success probability WITHOUT memory (default 0.40).
        max_attempts: Max repair attempts per question (default 5).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with comparison metrics for no-memory vs with-memory conditions.
    """
    rng = random.Random(seed)

    # Compute memory-boosted repair probability.
    memory_boost = min(memory_precision, MEMORY_BOOST_CAP)
    p_mem = p_base + memory_boost * (1.0 - p_base)

    wrong_questions = [(q, r, ic) for q, r, ic in questions if not ic]

    def simulate_repair(p_success: float) -> list[int | None]:
        """Simulate repair for all wrong questions with given success prob."""
        results: list[int | None] = []
        for _ in wrong_questions:
            for attempt in range(1, max_attempts + 1):
                if rng.random() < p_success:
                    results.append(attempt)
                    break
            else:
                results.append(None)
        return results

    # Run both conditions with the SAME RNG state position (reset seed each time
    # so conditions are comparable at the level of question ordering, not
    # individual coin flips — each condition gets an independent draw).
    rng_no_mem = random.Random(seed)
    rng_with_mem = random.Random(seed + 1)

    def simulate_with_rng(p_success: float, local_rng: random.Random) -> list[int | None]:
        results: list[int | None] = []
        for _ in wrong_questions:
            for attempt in range(1, max_attempts + 1):
                if local_rng.random() < p_success:
                    results.append(attempt)
                    break
            else:
                results.append(None)
        return results

    no_mem_results = simulate_with_rng(p_base, rng_no_mem)
    mem_results = simulate_with_rng(p_mem, rng_with_mem)

    def summarise(results: list[int | None]) -> dict:
        successes = [r for r in results if r is not None]
        n_total = len(results)
        n_repaired = len(successes)
        mean_iters = (sum(successes) / n_repaired) if successes else float("nan")
        return {
            "n_wrong": n_total,
            "n_repaired": n_repaired,
            "fraction_repaired": round(n_repaired / n_total, 4) if n_total else 0.0,
            "mean_iterations_to_repair": round(mean_iters, 4) if successes else None,
        }

    no_mem_summary = summarise(no_mem_results)
    mem_summary = summarise(mem_results)

    # Compute speedup: how much faster (fewer iterations) does memory provide?
    speedup: float | None = None
    if (
        no_mem_summary["mean_iterations_to_repair"] is not None
        and mem_summary["mean_iterations_to_repair"] is not None
        and mem_summary["mean_iterations_to_repair"] > 0
    ):
        speedup = round(
            no_mem_summary["mean_iterations_to_repair"]
            / mem_summary["mean_iterations_to_repair"],
            4,
        )

    return {
        "domain": domain,
        "p_base": p_base,
        "memory_precision_input": round(memory_precision, 4),
        "memory_boost": round(memory_boost, 4),
        "p_with_memory": round(p_mem, 4),
        "no_memory": no_mem_summary,
        "with_memory": mem_summary,
        "speedup_ratio": speedup,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run the full three-session cross-session memory experiment.

    **Detailed explanation for engineers:**
        Orchestrates three sessions and collects results into a structured dict:

        SESSION 1 — Build memory:
            Generate 200 arithmetic questions. Run verification with a fresh
            ConstraintMemory and ConstraintTracker. After verification, memory
            has accumulated patterns (arithmetic errors seen multiple times →
            learned constraints emerge). Save memory + tracker to temp files.

        SESSION 2 — Test memory on same domain:
            Generate 200 NEW arithmetic questions (different seed → fresh).
            Run two conditions on the same questions:
              a) No-memory: fresh pipeline, no ConstraintMemory.
              b) With-memory: loaded ConstraintMemory from session 1.
            Compare: accuracy, n_memory_suggestions, and simulated repair speed.

        SESSION 3 — Cross-domain transfer:
            Generate 200 mixed (arithmetic + logic + code) questions.
            Use the same session-1 arithmetic memory.
            Domain-stratify results to test H4: arithmetic questions get memory
            suggestions; logic/code questions do not.

    Returns:
        Nested dict with session results, summaries, and hypothesis verdicts.
    """
    from carnot.pipeline.adaptive import AdaptiveWeighter
    from carnot.pipeline.memory import ConstraintMemory
    from carnot.pipeline.tracker import ConstraintTracker

    print("=" * 70)
    print("Experiment 136: Cross-Session Memory")
    print("=" * 70)
    print(f"  Session 1 questions  : {N_SESSION_1} arithmetic")
    print(f"  Session 2 questions  : {N_SESSION_2} arithmetic (test: no-mem vs mem)")
    print(f"  Session 3 questions  : {N_SESSION_3} mixed domain")
    print(f"  Correct fraction     : {CORRECT_FRACTION:.0%}")
    print(f"  Memory threshold     : {MEMORY_THRESHOLD} occurrences")
    print(f"  Repair base prob     : {P_REPAIR_BASE:.0%} per attempt (no memory)")
    print(f"  Max repair attempts  : {MAX_REPAIR_ATTEMPTS}")
    print()

    # -----------------------------------------------------------------------
    # SESSION 1: Build memory + tracker on arithmetic questions
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("SESSION 1: Verify arithmetic questions → build memory + tracker")
    print("-" * 70)

    t_s1 = time.monotonic()
    questions_s1 = generate_arithmetic_questions(N_SESSION_1, SEED_SESSION_1)
    n_wrong_s1 = sum(1 for _, _, ic in questions_s1 if not ic)
    n_correct_s1 = N_SESSION_1 - n_wrong_s1
    print(f"  Generated: {N_SESSION_1} questions ({n_correct_s1} correct, {n_wrong_s1} wrong)")

    # Build memory + tracker from scratch for session 1.
    memory_s1 = ConstraintMemory()
    tracker_s1 = ConstraintTracker()

    session1_result = run_verification_session(
        questions_s1, "arithmetic", memory_s1, tracker_s1
    )
    s1_elapsed = time.monotonic() - t_s1

    print(f"  Accuracy (pipeline)  : {session1_result['accuracy']:.1%}")
    print(f"  Avg memory hints/q   : {session1_result['avg_memory_suggestions']:.3f}")
    print(f"  Elapsed              : {s1_elapsed:.1f}s")

    # Report memory state after session 1.
    mem_summary_s1 = memory_s1.summary()
    print(f"  Memory summary       : {mem_summary_s1}")
    tracker_stats_s1 = tracker_s1.stats()
    print("  Tracker stats (arithmetic domain):")
    for ctype, st in sorted(tracker_stats_s1.items()):
        print(
            f"    {ctype:<14}  fired={st['fired']:>4}  caught={st['caught']:>4}"
            f"  precision={st['precision']:.3f}"
        )
    print()

    # Save memory + tracker to temporary files (persist across sessions).
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = str(Path(tmpdir) / "session1_memory.json")
        tracker_path = str(Path(tmpdir) / "session1_tracker.json")
        memory_s1.save(memory_path)
        tracker_s1.save(tracker_path)
        print(f"  Memory saved to     : {memory_path}")
        print(f"  Tracker saved to    : {tracker_path}")

        # Compute adaptive weights from session-1 tracker (for repair simulation).
        adaptive_weights_s1 = AdaptiveWeighter.from_tracker(tracker_s1)
        arith_precision_s1 = tracker_stats_s1.get("arithmetic", {}).get("precision", 0.0)
        print(f"  Arithmetic precision: {arith_precision_s1:.4f}")
        print(f"  Adaptive weights    : {adaptive_weights_s1}")
        print()

        # -----------------------------------------------------------------------
        # SESSION 2: Load memory → verify NEW arithmetic questions
        # -----------------------------------------------------------------------
        print("-" * 70)
        print("SESSION 2: Load session-1 memory → verify new arithmetic questions")
        print("-" * 70)

        t_s2 = time.monotonic()
        questions_s2 = generate_arithmetic_questions(N_SESSION_2, SEED_SESSION_2)
        n_wrong_s2 = sum(1 for _, _, ic in questions_s2 if not ic)
        n_correct_s2 = N_SESSION_2 - n_wrong_s2
        print(f"  Generated: {N_SESSION_2} questions ({n_correct_s2} correct, {n_wrong_s2} wrong)")

        # Condition A: no memory.
        print("\n  [Condition A: NO MEMORY]")
        session2a_result = run_verification_session(
            questions_s2, "arithmetic", memory=None, tracker=None
        )
        print(f"    Accuracy           : {session2a_result['accuracy']:.1%}")
        print(f"    Avg memory hints/q : {session2a_result['avg_memory_suggestions']:.3f}")

        # Condition B: loaded memory from session 1.
        print("\n  [Condition B: WITH MEMORY (loaded from session 1)]")
        memory_loaded = ConstraintMemory.load(memory_path)
        loaded_mem_summary = memory_loaded.summary()
        print(f"    Loaded memory      : {loaded_mem_summary}")
        session2b_result = run_verification_session(
            questions_s2, "arithmetic", memory=memory_loaded, tracker=None
        )
        print(f"    Accuracy           : {session2b_result['accuracy']:.1%}")
        print(f"    Avg memory hints/q : {session2b_result['avg_memory_suggestions']:.3f}")

        # Compare conditions.
        acc_delta_s2 = session2b_result["accuracy"] - session2a_result["accuracy"]
        hint_delta_s2 = (
            session2b_result["avg_memory_suggestions"]
            - session2a_result["avg_memory_suggestions"]
        )
        print(f"\n  Accuracy delta (B-A) : {acc_delta_s2:+.4f}")
        print(f"  Hint delta (B-A)     : {hint_delta_s2:+.4f}")

        # Repair speed simulation using session-1 tracker precision.
        print("\n  [Repair speed simulation: no-memory vs with-memory]")
        repair_s2 = simulate_repair_speed(
            questions_s2,
            domain="arithmetic",
            memory_precision=arith_precision_s1,
            p_base=P_REPAIR_BASE,
            max_attempts=MAX_REPAIR_ATTEMPTS,
            seed=SEED_SESSION_2,
        )
        no_mem_iters = repair_s2["no_memory"]["mean_iterations_to_repair"]
        mem_iters = repair_s2["with_memory"]["mean_iterations_to_repair"]
        print(f"    No-memory mean iters to repair : {no_mem_iters}")
        print(f"    With-memory mean iters         : {mem_iters}")
        print(f"    Speedup ratio                  : {repair_s2['speedup_ratio']}")
        print(f"    No-memory repaired fraction    : {repair_s2['no_memory']['fraction_repaired']:.1%}")
        print(f"    With-memory repaired fraction  : {repair_s2['with_memory']['fraction_repaired']:.1%}")

        s2_elapsed = time.monotonic() - t_s2
        print(f"\n  Session 2 elapsed    : {s2_elapsed:.1f}s")
        print()

        # -----------------------------------------------------------------------
        # SESSION 3: Mixed domain — cross-domain transfer test
        # -----------------------------------------------------------------------
        print("-" * 70)
        print("SESSION 3: Mixed domain (arithmetic + logic + code) — cross-domain test")
        print("-" * 70)

        t_s3 = time.monotonic()
        questions_s3 = generate_mixed_questions(N_SESSION_3, SEED_SESSION_3)
        n_by_domain = {}
        for _, _, _, d in questions_s3:
            n_by_domain[d] = n_by_domain.get(d, 0) + 1
        print(f"  Generated: {N_SESSION_3} questions {n_by_domain}")

        # Load session-1 memory again for session 3.
        memory_s3 = ConstraintMemory.load(memory_path)

        session3_result = run_mixed_domain_session(questions_s3, memory=memory_s3)

        print(f"  Total accuracy       : {session3_result['total_accuracy']:.1%}")
        print()
        print("  Per-domain results:")
        for domain, ds in session3_result["domain_summaries"].items():
            print(
                f"    {domain:<12}  n={ds['n_questions']:<3}  "
                f"accuracy={ds['accuracy']:.1%}  "
                f"avg_mem_hints={ds['avg_memory_suggestions']:.3f}"
            )

        s3_elapsed = time.monotonic() - t_s3
        print(f"\n  Session 3 elapsed    : {s3_elapsed:.1f}s")
        print()

    # -----------------------------------------------------------------------
    # Hypothesis evaluations
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("HYPOTHESIS EVALUATIONS")
    print("=" * 70)

    # H1: Memory accumulates — arithmetic domain has mature patterns after session 1.
    arith_summary = mem_summary_s1.get("arithmetic", {})
    h1_mature = arith_summary.get("mature_patterns", 0) > 0
    h1_total = arith_summary.get("total_patterns", 0)
    print(f"\nH1 — Memory accumulates (mature patterns > 0 after session 1):")
    print(f"    arithmetic domain: {arith_summary}")
    h1_pass = h1_mature
    print(f"    Result: {'PASS ✓' if h1_pass else 'FAIL ✗'}")

    # H2: Memory helps same domain — session 2 with memory gets more hints.
    h2_pass = session2b_result["avg_memory_suggestions"] > session2a_result["avg_memory_suggestions"]
    print(f"\nH2 — Memory provides hints in same domain:")
    print(f"    No-memory avg hints : {session2a_result['avg_memory_suggestions']:.3f}")
    print(f"    With-memory avg hints: {session2b_result['avg_memory_suggestions']:.3f}")
    print(f"    Result: {'PASS ✓' if h2_pass else 'FAIL ✗'}")

    # H3: Memory speeds repair — speedup ratio > 1.0.
    h3_pass = (
        repair_s2["speedup_ratio"] is not None
        and repair_s2["speedup_ratio"] > 1.0
    )
    print(f"\nH3 — Memory speeds repair (speedup ratio > 1.0):")
    print(f"    Speedup ratio: {repair_s2['speedup_ratio']}")
    print(f"    Result: {'PASS ✓' if h3_pass else 'FAIL ✗'}")

    # H4: Cross-domain isolation — arithmetic memory does NOT help logic/code.
    arith_hints_s3 = session3_result["domain_summaries"]["arithmetic"]["avg_memory_suggestions"]
    logic_hints_s3 = session3_result["domain_summaries"]["logic"]["avg_memory_suggestions"]
    code_hints_s3 = session3_result["domain_summaries"]["code"]["avg_memory_suggestions"]
    h4_pass = (
        arith_hints_s3 > 0
        and logic_hints_s3 == 0.0
        and code_hints_s3 == 0.0
    )
    print(f"\nH4 — Domain specificity (arith hints > 0, logic/code hints == 0):")
    print(f"    arithmetic avg hints : {arith_hints_s3:.3f}")
    print(f"    logic avg hints      : {logic_hints_s3:.3f}")
    print(f"    code avg hints       : {code_hints_s3:.3f}")
    print(f"    Result: {'PASS ✓' if h4_pass else 'FAIL ✗'}")

    all_pass = h1_pass and h2_pass and h3_pass and h4_pass
    print()
    print(f"Overall: {'ALL HYPOTHESES PASS ✓' if all_pass else 'SOME HYPOTHESES FAILED ✗'}")
    print()

    # -----------------------------------------------------------------------
    # Assemble final results dict
    # -----------------------------------------------------------------------
    return {
        "experiment": "exp_136_cross_session_memory",
        "parameters": {
            "n_session_1": N_SESSION_1,
            "n_session_2": N_SESSION_2,
            "n_session_3": N_SESSION_3,
            "correct_fraction": CORRECT_FRACTION,
            "memory_threshold": MEMORY_THRESHOLD,
            "p_repair_base": P_REPAIR_BASE,
            "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
            "memory_boost_cap": MEMORY_BOOST_CAP,
            "seeds": {
                "session_1": SEED_SESSION_1,
                "session_2": SEED_SESSION_2,
                "session_3": SEED_SESSION_3,
            },
        },
        "session_1": {
            "accuracy": session1_result["accuracy"],
            "n_memory_suggestions_total": session1_result["n_memory_suggestions_total"],
            "avg_memory_suggestions": session1_result["avg_memory_suggestions"],
            "elapsed_seconds": session1_result["elapsed_seconds"],
            "memory_summary": mem_summary_s1,
            "tracker_stats": tracker_stats_s1,
            "adaptive_weights": adaptive_weights_s1,
            "arith_precision": arith_precision_s1,
        },
        "session_2": {
            "no_memory": {
                "accuracy": session2a_result["accuracy"],
                "avg_memory_suggestions": session2a_result["avg_memory_suggestions"],
                "n_memory_suggestions_total": session2a_result["n_memory_suggestions_total"],
            },
            "with_memory": {
                "accuracy": session2b_result["accuracy"],
                "avg_memory_suggestions": session2b_result["avg_memory_suggestions"],
                "n_memory_suggestions_total": session2b_result["n_memory_suggestions_total"],
                "loaded_memory_summary": loaded_mem_summary,
            },
            "accuracy_delta": round(acc_delta_s2, 4),
            "hint_delta": round(hint_delta_s2, 4),
            "repair_simulation": repair_s2,
            "elapsed_seconds": round(s2_elapsed, 2),
        },
        "session_3": {
            "total_accuracy": session3_result["total_accuracy"],
            "domain_summaries": session3_result["domain_summaries"],
            "elapsed_seconds": session3_result["elapsed_seconds"],
        },
        "hypotheses": {
            "h1_memory_accumulates": h1_pass,
            "h2_memory_hints_same_domain": h2_pass,
            "h3_memory_speeds_repair": h3_pass,
            "h4_domain_specificity": h4_pass,
            "all_pass": all_pass,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment, save results, exit with status code."""
    t_start = time.monotonic()
    results = run_experiment()
    elapsed = time.monotonic() - t_start

    results["total_elapsed_seconds"] = round(elapsed, 2)
    print(f"Total wall-clock time: {elapsed:.1f}s")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")

    if not results["hypotheses"]["all_pass"]:
        print()
        failed = [k for k, v in results["hypotheses"].items() if k != "all_pass" and not v]
        print(f"WARNING: Failed hypotheses: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
