"""Experiment 145: JEPA Fast-Path Integration Benchmark.

**Researcher summary:**
    Integrates the Tier 3 JEPA predictor (Exp 144) into VerifyRepairPipeline
    as an optional early-exit gate. Benchmarks three modes across 500 synthetic
    questions (200 arithmetic, 200 code, 100 logic):
        1. Baseline — no gating, full Ising verification for all
        2. JEPA gate at threshold=0.3 — aggressive fast-path
        3. JEPA gate at threshold=0.5 — conservative fast-path
    Reports: fast-path rate, per-path accuracy, overall accuracy vs baseline,
    wall-clock speedup, and error analysis of fast-path misses.

**Tier 3 architecture (research-program.md):**
    The pipeline now has two "lanes":
      - Slow lane (full): embed → extract constraints → Ising verify
      - Fast lane (JEPA): embed first 50 tokens → JEPA MLP → if max_prob < threshold,
        return verified=True immediately (skip expensive full verification)
    The fast lane is an optimistic gate: we assume low-risk responses are clean.
    Fast-path errors occur when the gate fires but the full pipeline would find
    a violation — these are "missed" violations.

**Results written to:**
    results/experiment_145_results.json

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_145_jepa_fastpath.py

Spec: REQ-JEPA-002, REQ-VERIFY-003, SCENARIO-JEPA-001
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure the package root is on the Python path when run from project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.embeddings.fast_embedding import RandomProjectionEmbedding
from carnot.pipeline.jepa_predictor import JEPAViolationPredictor
from carnot.pipeline.verify_repair import VerifyRepairPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PREDICTOR_PATH = Path("results/jepa_predictor.safetensors")
RESULTS_PATH = Path("results/experiment_145_results.json")

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

N_ARITHMETIC = 200
N_CODE = 200
N_LOGIC = 100
N_TOTAL = N_ARITHMETIC + N_CODE + N_LOGIC

THRESHOLDS = [0.3, 0.5]

# Target: ≥40% fast-path rate with <2% accuracy degradation.
TARGET_FASTPATH_RATE = 0.40
TARGET_MAX_DEGRADATION = 0.02

SEED = 42


# ---------------------------------------------------------------------------
# Synthetic Q&A generation
# ---------------------------------------------------------------------------


def generate_arithmetic_qa(rng: random.Random) -> list[dict]:
    """Generate 200 arithmetic Q&A pairs (mix of correct and incorrect).

    **Detailed explanation for engineers:**
        Each entry has a ``question`` (e.g., "What is 17 + 38?"), a ``response``
        (the answer text), a ``domain`` tag, and a ``ground_truth_correct`` flag
        indicating whether the response is arithmetically correct. Approximately
        60% of responses are correct so the dataset is not too unbalanced.

        This simulates the distribution of LLM responses where most answers are
        right but a meaningful fraction contains arithmetic errors — matching the
        Exp 143 training distribution (~33% violation rate).

    Args:
        rng: Random instance for reproducibility.

    Returns:
        List of Q&A dicts, each with keys: question, response, domain,
        ground_truth_correct.
    """
    pairs = []
    ops = [
        ("+", lambda a, b: a + b, "sum"),
        ("-", lambda a, b: a - b, "difference"),
        ("*", lambda a, b: a * b, "product"),
    ]
    for _ in range(N_ARITHMETIC):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op_sym, op_fn, op_name = rng.choice(ops)
        correct_result = op_fn(a, b)

        # ~60% correct, ~40% wrong (inject an off-by-one or ±20% error).
        if rng.random() < 0.60:
            answer = correct_result
            is_correct = True
        else:
            # Wrong answer: add a random offset to simulate an arithmetic error.
            offset = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5, 10, -10])
            answer = correct_result + offset
            is_correct = False

        question = f"What is {a} {op_sym} {b}?"
        response = (
            f"To compute {a} {op_sym} {b}, I {op_name} {a} and {b}. "
            f"The answer is {a} {op_sym} {b} = {answer}."
        )
        pairs.append(
            {
                "question": question,
                "response": response,
                "domain": "arithmetic",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


def generate_code_qa(rng: random.Random) -> list[dict]:
    """Generate 200 code Q&A pairs (Python snippets with and without bugs).

    **Detailed explanation for engineers:**
        Each entry is a simple Python coding question. Correct responses have
        syntactically valid, logically correct code. Incorrect responses have
        a deliberate bug (wrong variable name, off-by-one, undefined variable).
        We do NOT actually execute the code — correctness is determined by our
        synthetic ground truth flag.

        In the benchmark, the pipeline's code extractor checks for Python syntax
        errors and undefined-variable patterns. We expect the JEPA predictor to
        have lower AUROC on code than arithmetic (per Exp 144 training data), so
        code violations are harder to predict early.

    Args:
        rng: Random instance for reproducibility.

    Returns:
        List of Q&A dicts for code domain.
    """
    templates = [
        {
            "question": "Write Python code to check if {n} is even.",
            "correct": "def is_even(n):\n    return n % 2 == 0\n\nresult = is_even({n})\nprint(result)",
            "wrong": "def is_even(n):\n    return n % 2 = 0\n\nresult = is_even({n})\nprint(result)",
        },
        {
            "question": "Write Python code to compute the factorial of {n}.",
            "correct": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial({n})\nprint(result)",
            "wrong": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n)\n\nresult = factorial({n})\nprint(result)",
        },
        {
            "question": "Write Python code to reverse a list.",
            "correct": "def reverse_list(lst):\n    return lst[::-1]\n\nmy_list = [1, 2, 3, 4, 5]\nresult = reverse_list(my_list)\nprint(result)",
            "wrong": "def reverse_list(lst):\n    return lst.reverse()\n\nmy_list = [1, 2, 3, 4, 5]\nresult = reverse_list(my_list)\nprint(result)",
        },
        {
            "question": "Write Python code to sum a list of numbers.",
            "correct": "def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\nresult = sum_list([1, 2, 3, 4, 5])\nprint(result)",
            "wrong": "def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += nums\n    return total\n\nresult = sum_list([1, 2, 3, 4, 5])\nprint(result)",
        },
        {
            "question": "Write Python code to find the maximum in a list.",
            "correct": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)\n\nresult = find_max([3, 1, 4, 1, 5, 9])\nprint(result)",
            "wrong": "def find_max(lst):\n    if not lst:\n        return None\n    return maximum(lst)\n\nresult = find_max([3, 1, 4, 1, 5, 9])\nprint(result)",
        },
    ]

    pairs = []
    for i in range(N_CODE):
        tmpl = templates[i % len(templates)]
        n = rng.randint(3, 12)
        is_correct = rng.random() < 0.60
        code = (
            tmpl["correct"].format(n=n)
            if is_correct
            else tmpl["wrong"].format(n=n)
        )
        pairs.append(
            {
                "question": tmpl["question"].format(n=n),
                "response": f"Here is the Python code:\n\n```python\n{code}\n```",
                "domain": "code",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


def generate_logic_qa(rng: random.Random) -> list[dict]:
    """Generate 100 logic Q&A pairs (syllogisms and conditional reasoning).

    **Detailed explanation for engineers:**
        Each entry is a classic syllogism or modus ponens question. Correct
        responses draw valid inferences. Incorrect responses either negate the
        conclusion or commit a common logical fallacy (affirming the consequent).

        Logic constraints are the hardest for the JEPA predictor because the
        violation signal is semantic (does the conclusion follow?) rather than
        syntactic (is a number wrong?). This makes the fast-path less reliable
        for logic — which is expected and documented in the error analysis.

    Args:
        rng: Random instance for reproducibility.

    Returns:
        List of Q&A dicts for logic domain.
    """
    syllogisms = [
        ("All mammals are warm-blooded. Dogs are mammals.", "Dogs are warm-blooded.", "Dogs are not warm-blooded."),
        ("All birds have feathers. Eagles are birds.", "Eagles have feathers.", "Eagles do not have feathers."),
        ("If it rains, the ground gets wet. It is raining.", "The ground is wet.", "The ground is not wet."),
        ("All squares are rectangles. ABCD is a square.", "ABCD is a rectangle.", "ABCD is not a rectangle."),
        ("If P then Q. P is true.", "Q is true.", "Q is false."),
        ("All prime numbers greater than 2 are odd. 7 is prime and greater than 2.", "7 is odd.", "7 is even."),
        ("All cats are felines. Whiskers is a cat.", "Whiskers is a feline.", "Whiskers is not a feline."),
        ("If the battery is charged, the device works. The battery is charged.", "The device works.", "The device does not work."),
        ("All integers divisible by 4 are divisible by 2. 12 is divisible by 4.", "12 is divisible by 2.", "12 is not divisible by 2."),
        ("All roses are flowers. Some flowers fade quickly. Some roses are red.", "Roses are a type of flower.", "Roses are not flowers."),
    ]

    pairs = []
    for i in range(N_LOGIC):
        premises, correct_conclusion, wrong_conclusion = syllogisms[i % len(syllogisms)]
        is_correct = rng.random() < 0.60
        conclusion = correct_conclusion if is_correct else wrong_conclusion

        question = f"{premises} What can we conclude?"
        response = (
            f"Given: {premises}\n"
            f"By logical deduction, we can conclude: {conclusion}\n"
            f"Therefore, the answer is: {conclusion}"
        )
        pairs.append(
            {
                "question": question,
                "response": response,
                "domain": "logic",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    """Results for one benchmarking mode (baseline or a threshold).

    Attributes:
        name: Human-readable mode name (e.g., "baseline", "threshold=0.3").
        fast_path_count: Number of questions that took the JEPA fast path.
        slow_path_count: Number of questions that ran full verification.
        fast_path_correct: Count of fast-path questions where decision was right.
        slow_path_correct: Count of slow-path questions where decision was right.
        fast_path_errors: List of fast-path errors (dicts with question details).
        total_wall_time_s: Total wall-clock seconds for all 500 questions.
        per_question_times_s: List of per-question wall-clock times.
    """

    name: str
    fast_path_count: int = 0
    slow_path_count: int = 0
    fast_path_correct: int = 0
    slow_path_correct: int = 0
    fast_path_errors: list = None  # type: ignore[assignment]
    total_wall_time_s: float = 0.0
    per_question_times_s: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fast_path_errors is None:
            self.fast_path_errors = []
        if self.per_question_times_s is None:
            self.per_question_times_s = []

    @property
    def fast_path_rate(self) -> float:
        """Fraction of questions that took the fast path (0.0–1.0)."""
        total = self.fast_path_count + self.slow_path_count
        return self.fast_path_count / max(total, 1)

    @property
    def fast_path_accuracy(self) -> float:
        """Accuracy on fast-path questions: fraction where decision was correct."""
        return self.fast_path_correct / max(self.fast_path_count, 1)

    @property
    def slow_path_accuracy(self) -> float:
        """Accuracy on slow-path questions (full pipeline always matches ground truth
        within the definition of correctness used here)."""
        return self.slow_path_correct / max(self.slow_path_count, 1)

    @property
    def overall_accuracy(self) -> float:
        """Fraction of all 500 questions where our decision matched ground truth."""
        total_correct = self.fast_path_correct + self.slow_path_correct
        total = self.fast_path_count + self.slow_path_count
        return total_correct / max(total, 1)


def run_benchmark(
    qa_pairs: list[dict],
    pipeline: VerifyRepairPipeline,
    predictor: JEPAViolationPredictor | None,
    threshold: float,
    mode_name: str,
    baseline_decisions: list[bool] | None = None,
) -> ModeResult:
    """Run one benchmark mode over all Q&A pairs.

    **Detailed explanation for engineers:**
        For each question, calls ``pipeline.verify()`` with or without the JEPA
        predictor. Times each call individually.

        "Correct decision" is defined relative to the baseline (no-gating) pass:
        - If baseline says verified=True and our decision is verified=True → correct
        - If baseline says verified=False and our decision is verified=False → correct
        - Any mismatch → wrong

        For the baseline mode itself, "correct" is defined by ground_truth_correct
        (whether the response was synthetically generated to be correct or not).
        Note: the pipeline's extractor may not catch all synthetic errors (it's a
        real extractor, not an oracle), so baseline accuracy may differ from
        ground-truth accuracy — this is expected and documented.

    Args:
        qa_pairs: List of Q&A dicts from generate_*_qa().
        pipeline: VerifyRepairPipeline instance.
        predictor: JEPAViolationPredictor, or None for baseline (no gating).
        threshold: JEPA threshold to use (ignored when predictor is None).
        mode_name: Label for logging and results.
        baseline_decisions: List of baseline verified flags (one per qa_pair).
            If None, ground_truth_correct is used as the reference instead.

    Returns:
        ModeResult with all collected metrics.
    """
    result = ModeResult(name=mode_name)
    t_total_start = time.perf_counter()

    for i, qa in enumerate(qa_pairs):
        t_q_start = time.perf_counter()

        vr = pipeline.verify(
            question=qa["question"],
            response=qa["response"],
            domain=qa["domain"],
            jepa_predictor=predictor,
            jepa_threshold=threshold,
        )

        t_q_end = time.perf_counter()
        result.per_question_times_s.append(t_q_end - t_q_start)

        took_fast_path = vr.skipped

        # Determine the reference "correct" decision.
        if baseline_decisions is not None:
            # Compare to baseline: correct = decision matches baseline.
            reference_verified = baseline_decisions[i]
        else:
            # For baseline itself: compare to pipeline's own result (always matches).
            # We record ground_truth_correct as a separate field for analysis.
            reference_verified = vr.verified

        decision_correct = (vr.verified == reference_verified)

        if took_fast_path:
            result.fast_path_count += 1
            if decision_correct:
                result.fast_path_correct += 1
            else:
                # Record details for error analysis.
                result.fast_path_errors.append(
                    {
                        "index": i,
                        "domain": qa["domain"],
                        "question": qa["question"][:100],
                        "response_snippet": qa["response"][:80],
                        "ground_truth_correct": qa["ground_truth_correct"],
                        "our_decision": vr.verified,
                        "reference_decision": reference_verified,
                        "jepa_probs": vr.certificate.get("jepa_probs", {}),
                        "jepa_max_prob": vr.certificate.get("jepa_max_prob", 0.0),
                    }
                )
        else:
            result.slow_path_count += 1
            if decision_correct:
                result.slow_path_correct += 1

    t_total_end = time.perf_counter()
    result.total_wall_time_s = t_total_end - t_total_start

    return result


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def analyze_errors(errors: list[dict], mode_name: str) -> dict:
    """Analyze fast-path errors by domain and response characteristics.

    **Detailed explanation for engineers:**
        Fast-path errors occur when:
        - JEPA predicted "low risk" (max_prob < threshold)
        - But the full pipeline would have found a violation (reference_decision=False)
        - So our fast-path returned verified=True incorrectly

        We analyze: which domains have the most errors, whether short responses
        (≤ 50 whitespace tokens) are overrepresented (hypothesis: short responses
        hit the first-50-token window fully, so the JEPA signal might be worse),
        and the JEPA max-probability distribution for errors vs correct fast-paths.

    Args:
        errors: List of error dicts from ModeResult.fast_path_errors.
        mode_name: Label for the mode being analyzed.

    Returns:
        Dict with domain_counts, avg_jepa_max_prob, short_response_fraction.
    """
    if not errors:
        return {
            "mode": mode_name,
            "n_errors": 0,
            "domain_counts": {},
            "avg_jepa_max_prob": None,
            "short_response_fraction": None,
        }

    domain_counts: dict[str, int] = {}
    jepa_probs = []
    short_count = 0

    for err in errors:
        domain = err["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        jepa_probs.append(err["jepa_max_prob"])
        # "Short response": response is 50 tokens or fewer (fully in first-50 window).
        n_tokens = len(err["response_snippet"].split())
        if n_tokens <= 50:
            short_count += 1

    return {
        "mode": mode_name,
        "n_errors": len(errors),
        "domain_counts": domain_counts,
        "avg_jepa_max_prob": float(np.mean(jepa_probs)) if jepa_probs else None,
        "short_response_fraction": short_count / len(errors),
        "most_common_domain": max(domain_counts, key=lambda d: domain_counts[d]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 145: JEPA fast-path integration benchmark."""
    rng = random.Random(SEED)

    print("=" * 70)
    print("Experiment 145: JEPA Fast-Path Integration Benchmark")
    print("=" * 70)

    # --- Load JEPA predictor ---
    if not PREDICTOR_PATH.exists():
        print(f"ERROR: JEPA predictor not found at {PREDICTOR_PATH}")
        print("Run experiment_144_train_jepa.py first to produce the model.")
        sys.exit(1)

    print(f"\nLoading JEPA predictor from {PREDICTOR_PATH}...")
    predictor = JEPAViolationPredictor()
    predictor.load(str(PREDICTOR_PATH))
    print("  Predictor loaded.")

    # --- Generate synthetic Q&A pairs ---
    print(f"\nGenerating {N_TOTAL} synthetic Q&A pairs...")
    arithmetic_pairs = generate_arithmetic_qa(rng)
    code_pairs = generate_code_qa(rng)
    logic_pairs = generate_logic_qa(rng)
    all_pairs = arithmetic_pairs + code_pairs + logic_pairs

    n_correct = sum(1 for p in all_pairs if p["ground_truth_correct"])
    print(
        f"  {N_ARITHMETIC} arithmetic, {N_CODE} code, {N_LOGIC} logic  "
        f"({n_correct}/{N_TOTAL} = {100*n_correct/N_TOTAL:.1f}% ground-truth correct)"
    )

    # --- Set up pipeline (no LLM, verify-only mode) ---
    pipeline = VerifyRepairPipeline(timeout_seconds=10.0)

    # --- Mode 1: Baseline (no gating) ---
    print("\n--- Mode 1: Baseline (no JEPA gating) ---")
    baseline_result = run_benchmark(
        qa_pairs=all_pairs,
        pipeline=pipeline,
        predictor=None,
        threshold=0.5,  # ignored
        mode_name="baseline",
        baseline_decisions=None,
    )
    # Baseline's "decisions" for comparison purposes.
    baseline_decisions = []
    for qa in all_pairs:
        vr = pipeline.verify(question=qa["question"], response=qa["response"], domain=qa["domain"])
        baseline_decisions.append(vr.verified)

    n_baseline_verified = sum(baseline_decisions)
    print(f"  Total time: {baseline_result.total_wall_time_s:.2f}s")
    print(f"  Fast-path rate: 0.0% (all slow-path)")
    print(f"  Baseline verified: {n_baseline_verified}/{N_TOTAL} = {100*n_baseline_verified/N_TOTAL:.1f}%")
    # Accuracy vs ground truth.
    baseline_gt_match = sum(
        1 for i, qa in enumerate(all_pairs)
        if (baseline_decisions[i] == qa["ground_truth_correct"])
    )
    print(f"  Baseline accuracy vs ground truth: {100*baseline_gt_match/N_TOTAL:.1f}%")

    # --- Mode 2 & 3: JEPA gating at each threshold ---
    gating_results = []
    for threshold in THRESHOLDS:
        mode_name = f"threshold={threshold}"
        print(f"\n--- Mode: JEPA gate at threshold={threshold} ---")
        mode_result = run_benchmark(
            qa_pairs=all_pairs,
            pipeline=pipeline,
            predictor=predictor,
            threshold=threshold,
            mode_name=mode_name,
            baseline_decisions=baseline_decisions,
        )
        gating_results.append(mode_result)

        speedup = baseline_result.total_wall_time_s / max(mode_result.total_wall_time_s, 1e-9)
        print(f"  Total time: {mode_result.total_wall_time_s:.2f}s  (speedup: {speedup:.2f}×)")
        print(f"  Fast-path rate: {100*mode_result.fast_path_rate:.1f}%  (target ≥ {100*TARGET_FASTPATH_RATE:.0f}%)")
        print(f"  Fast-path accuracy: {100*mode_result.fast_path_accuracy:.1f}%")
        print(f"  Slow-path accuracy: {100*mode_result.slow_path_accuracy:.1f}%")

        baseline_accuracy = sum(
            1 for i in range(N_TOTAL)
            if (baseline_decisions[i] == baseline_decisions[i])
        ) / N_TOTAL  # baseline always "correct" vs itself = 1.0
        degradation = 1.0 - mode_result.overall_accuracy
        print(f"  Overall accuracy vs baseline: {100*mode_result.overall_accuracy:.1f}%")
        print(f"  Accuracy degradation: {100*degradation:.2f}%  (target < {100*TARGET_MAX_DEGRADATION:.0f}%)")
        print(f"  Fast-path errors: {len(mode_result.fast_path_errors)}")

        # Target check.
        fp_ok = mode_result.fast_path_rate >= TARGET_FASTPATH_RATE
        deg_ok = degradation <= TARGET_MAX_DEGRADATION
        status = "PASS" if (fp_ok and deg_ok) else "MISS"
        print(f"  Target [{TARGET_FASTPATH_RATE*100:.0f}% fast-path, <{TARGET_MAX_DEGRADATION*100:.0f}% degradation]: {status}")

    # --- Error analysis ---
    print("\n--- Error Analysis (fast-path misses) ---")
    error_analyses = []
    for mode_result in gating_results:
        analysis = analyze_errors(mode_result.fast_path_errors, mode_result.name)
        error_analyses.append(analysis)

        print(f"\nMode: {mode_result.name}")
        print(f"  Total fast-path errors: {analysis['n_errors']}")
        if analysis["n_errors"] > 0:
            print(f"  Errors by domain: {analysis['domain_counts']}")
            print(f"  Most common error domain: {analysis.get('most_common_domain', 'N/A')}")
            print(f"  Avg JEPA max_prob at error: {analysis['avg_jepa_max_prob']:.3f}")
            print(f"  Short-response errors (≤50 tokens): {100*analysis['short_response_fraction']:.1f}%")
            print("  Pattern analysis:")
            if analysis.get("most_common_domain") == "arithmetic":
                print("    → Arithmetic errors dominate: numeric patterns may be harder to detect")
                print("      from short prefixes before the final numeric result appears.")
            elif analysis.get("most_common_domain") == "logic":
                print("    → Logic errors dominate: semantic inference checking is harder to")
                print("      predict from byte-level embeddings (expected — JEPA trained mainly")
                print("      on arithmetic pairs per Exp 143 data distribution).")
            elif analysis.get("most_common_domain") == "code":
                print("    → Code errors dominate: syntax bugs may appear late in the response")
                print("      (after the first 50 tokens), so the prefix embedding misses them.")
            if analysis["short_response_fraction"] > 0.5:
                print("    → >50% of errors are from short responses — short answers route")
                print("      fully through the 50-token window, suggesting the predictor")
                print("      underweights confidence on minimal responses.")

    # --- Speedup summary ---
    print("\n--- Wall-Clock Speedup Summary ---")
    print(f"  Baseline total: {baseline_result.total_wall_time_s:.3f}s ({N_TOTAL} questions)")
    for mode_result in gating_results:
        speedup = baseline_result.total_wall_time_s / max(mode_result.total_wall_time_s, 1e-9)
        print(
            f"  {mode_result.name}: {mode_result.total_wall_time_s:.3f}s  "
            f"(speedup {speedup:.2f}×, {100*mode_result.fast_path_rate:.1f}% fast-path)"
        )

    # --- Save results ---
    results = {
        "experiment": 145,
        "description": "JEPA fast-path integration benchmark",
        "n_total": N_TOTAL,
        "n_arithmetic": N_ARITHMETIC,
        "n_code": N_CODE,
        "n_logic": N_LOGIC,
        "targets": {
            "fast_path_rate_min": TARGET_FASTPATH_RATE,
            "max_accuracy_degradation": TARGET_MAX_DEGRADATION,
        },
        "baseline": {
            "total_time_s": baseline_result.total_wall_time_s,
            "n_verified": n_baseline_verified,
            "gt_accuracy": baseline_gt_match / N_TOTAL,
        },
        "modes": [],
        "error_analysis": error_analyses,
    }

    for i, (threshold, mode_result) in enumerate(zip(THRESHOLDS, gating_results)):
        speedup = baseline_result.total_wall_time_s / max(mode_result.total_wall_time_s, 1e-9)
        degradation = 1.0 - mode_result.overall_accuracy
        results["modes"].append(
            {
                "name": mode_result.name,
                "threshold": threshold,
                "total_time_s": mode_result.total_wall_time_s,
                "speedup": speedup,
                "fast_path_count": mode_result.fast_path_count,
                "slow_path_count": mode_result.slow_path_count,
                "fast_path_rate": mode_result.fast_path_rate,
                "fast_path_accuracy": mode_result.fast_path_accuracy,
                "slow_path_accuracy": mode_result.slow_path_accuracy,
                "overall_accuracy": mode_result.overall_accuracy,
                "accuracy_degradation": degradation,
                "n_fast_path_errors": len(mode_result.fast_path_errors),
                "target_fast_path_met": mode_result.fast_path_rate >= TARGET_FASTPATH_RATE,
                "target_degradation_met": degradation <= TARGET_MAX_DEGRADATION,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode_result in gating_results:
        fp_rate = mode_result.fast_path_rate
        degradation = 1.0 - mode_result.overall_accuracy
        speedup = baseline_result.total_wall_time_s / max(mode_result.total_wall_time_s, 1e-9)
        fp_target = "✓" if fp_rate >= TARGET_FASTPATH_RATE else "✗"
        deg_target = "✓" if degradation <= TARGET_MAX_DEGRADATION else "✗"
        print(
            f"  {mode_result.name:20s}  "
            f"fast-path: {100*fp_rate:.1f}% {fp_target}  "
            f"degradation: {100*degradation:.2f}% {deg_target}  "
            f"speedup: {speedup:.2f}×"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
