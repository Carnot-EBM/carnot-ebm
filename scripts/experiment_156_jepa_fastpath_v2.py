"""Experiment 156: JEPA Fast-Path v2 Validation Benchmark.

**Researcher summary:**
    Validates the improved Tier 3 JEPA predictor (v2, trained in Exp 155 with
    multi-domain data) against the Exp 145 baseline. Exp 145 showed v1 had
    zero signal on code/logic domains, producing 19.8% accuracy degradation at
    threshold=0.5. v2 was retrained on multi-domain data targeting macro AUROC
    >0.70. This experiment determines whether v2 meets the fast-path targets.

**Goals (from research-program.md Tier 3):**
    - Find a threshold where fast_path_rate ≥ 40% AND accuracy_degradation < 2%
    - Per-domain: verify code/logic fast-path accuracy improves over v1
    - Wall-clock: measure overhead of the JEPA MLP gate vs time saved

**Benchmark setup:**
    - Same 500 synthetic Q&A pairs as Exp 145 (200 arithmetic, 200 code, 100 logic)
    - Same random seed (42) for exact reproducibility
    - Three JEPA thresholds: 0.3, 0.5, 0.7
    - Also runs with v1 predictor for direct comparison

**Target models referenced in task brief:**
    Qwen3.5-0.8B, google/gemma-4-E4B-it (not loaded here — benchmark uses
    synthetic Q&A pairs to isolate the JEPA gate effect from LLM quality).

**Results written to:**
    results/experiment_156_results.json

**v1 baseline numbers (from Exp 145) to beat:**
    threshold=0.3: fast_path_rate=38.0%, degradation=11.6%
    threshold=0.5: fast_path_rate=95.4%, degradation=19.8%
    Neither threshold met both targets simultaneously.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_156_jepa_fastpath_v2.py

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

V1_PREDICTOR_PATH = Path("results/jepa_predictor.safetensors")
V2_PREDICTOR_PATH = Path("results/jepa_predictor_v2.safetensors")
V1_RESULTS_PATH = Path("results/experiment_145_results.json")
RESULTS_PATH = Path("results/experiment_156_results.json")

# ---------------------------------------------------------------------------
# Benchmark configuration — identical to Exp 145 for reproducibility
# ---------------------------------------------------------------------------

N_ARITHMETIC = 200
N_CODE = 200
N_LOGIC = 100
N_TOTAL = N_ARITHMETIC + N_CODE + N_LOGIC

# Exp 145 used [0.3, 0.5]. Exp 156 adds 0.7 to find a better operating point.
THRESHOLDS = [0.3, 0.5, 0.7]

# Target from research-program.md Tier 3.
TARGET_FASTPATH_RATE = 0.40
TARGET_MAX_DEGRADATION = 0.02

# Must match Exp 145 exactly for reproducible Q&A pairs.
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic Q&A generation — exact copy from Exp 145 for reproducibility
# ---------------------------------------------------------------------------


def generate_arithmetic_qa(rng: random.Random) -> list[dict]:
    """Generate 200 arithmetic Q&A pairs (mix of correct and incorrect).

    **Detailed explanation for engineers:**
        Arithmetic Q&A pairs are simple word problems: "What is A op B?" with
        a response that states the answer. ~60% are correct; ~40% have a small
        offset error injected to simulate LLM arithmetic mistakes.

        This generator is seeded and deterministic — the same rng state from
        Exp 145 produces the same 200 questions. This is critical for
        comparing v1 vs v2 on the EXACT same inputs.

    Args:
        rng: Random instance already seeded with SEED=42.

    Returns:
        List of Q&A dicts with keys: question, response, domain,
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
        Code Q&A uses five Python snippet templates. Correct responses have
        valid, logically correct code. Incorrect responses have deliberate bugs
        (wrong operator, undefined variable, infinite recursion). v1 had near-
        zero signal on this domain; v2 was retrained with code examples to
        improve AUROC from ~0.50 to ~0.77 (per Exp 155 training results).

    Args:
        rng: Random instance already seeded with SEED=42.

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
        Logic Q&A uses 10 syllogism templates (modus ponens, categorical
        syllogisms). Correct responses state the valid conclusion; incorrect
        responses negate it. Logic was the weakest domain for both v1 and v2
        (per Exp 155: AUROC 0.534 v1, 0.479 v2 — v2 actually regressed slightly).
        This test validates whether the v2 regression on logic affects fast-path
        accuracy at the operating thresholds.

    Args:
        rng: Random instance already seeded with SEED=42.

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
# Benchmark dataclass — same structure as Exp 145 ModeResult
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    """Results for one benchmark mode (baseline or a JEPA threshold + predictor version).

    Attributes:
        name: Human-readable mode name (e.g., "v2_threshold=0.5").
        fast_path_count: Questions that took the JEPA fast path (skipped full verify).
        slow_path_count: Questions that ran full Ising verification.
        fast_path_correct: Fast-path questions where decision matched baseline.
        slow_path_correct: Slow-path questions where decision matched baseline.
        fast_path_errors: List of error detail dicts for analysis.
        total_wall_time_s: Total wall-clock seconds for all 500 questions.
        per_question_times_s: Per-question wall-clock times for latency analysis.
        per_domain_fast: Count of fast-path questions per domain.
        per_domain_correct: Count of correct fast-path decisions per domain.
    """

    name: str
    fast_path_count: int = 0
    slow_path_count: int = 0
    fast_path_correct: int = 0
    slow_path_correct: int = 0
    fast_path_errors: list = None  # type: ignore[assignment]
    total_wall_time_s: float = 0.0
    per_question_times_s: list = None  # type: ignore[assignment]
    per_domain_fast: dict = None  # type: ignore[assignment]
    per_domain_correct: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fast_path_errors is None:
            self.fast_path_errors = []
        if self.per_question_times_s is None:
            self.per_question_times_s = []
        if self.per_domain_fast is None:
            self.per_domain_fast = {"arithmetic": 0, "code": 0, "logic": 0}
        if self.per_domain_correct is None:
            self.per_domain_correct = {"arithmetic": 0, "code": 0, "logic": 0}

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
        """Accuracy on slow-path questions (full pipeline always matches baseline)."""
        return self.slow_path_correct / max(self.slow_path_count, 1)

    @property
    def overall_accuracy(self) -> float:
        """Fraction of all 500 questions where our decision matched baseline."""
        total_correct = self.fast_path_correct + self.slow_path_correct
        total = self.fast_path_count + self.slow_path_count
        return total_correct / max(total, 1)

    def per_domain_fast_path_accuracy(self) -> dict[str, float]:
        """Per-domain fast-path accuracy: how often did the fast path get it right?

        **Detailed explanation for engineers:**
            For each domain (arithmetic, code, logic), we compute what fraction
            of the fast-path decisions in that domain matched the baseline
            pipeline's full verification result. This is the key metric for
            assessing v2's improvement on code/logic where v1 was weakest.

        Returns:
            Dict mapping domain → accuracy in [0, 1], or NaN if no fast-path
            questions were routed from that domain.
        """
        result: dict[str, float] = {}
        for domain in ("arithmetic", "code", "logic"):
            n_fast = self.per_domain_fast[domain]
            n_correct = self.per_domain_correct[domain]
            result[domain] = n_correct / max(n_fast, 1) if n_fast > 0 else float("nan")
        return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


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
        Identical structure to Exp 145's run_benchmark() for comparability.
        For each question:
        - Calls pipeline.verify() with the given predictor and threshold.
        - Records whether it took the fast path (vr.skipped=True).
        - Compares the result to baseline_decisions[] (for gating modes) or
          vr.verified (for the baseline itself).
        - Tracks per-domain fast-path counts and accuracy.

        The wall-clock timing includes the JEPA MLP forward pass overhead for
        gating modes. In gating modes that skip full verification, the MLP
        overhead is the ONLY cost — so speedup requires fast_path_rate > 0 and
        MLP < full_pipeline_cost.

    Args:
        qa_pairs: 500 Q&A dicts from generate_*_qa().
        pipeline: VerifyRepairPipeline instance (shared across modes).
        predictor: JEPAViolationPredictor (v1 or v2), or None for baseline.
        threshold: JEPA threshold for fast-path gate.
        mode_name: Label for logging and results.
        baseline_decisions: Per-question baseline verified flags. If None,
            uses pipeline's own result as the reference (baseline mode).

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
        domain = qa["domain"]

        # Determine the reference "correct" decision.
        if baseline_decisions is not None:
            reference_verified = baseline_decisions[i]
        else:
            # Baseline mode: the pipeline's own result is the reference.
            reference_verified = vr.verified

        decision_correct = vr.verified == reference_verified

        if took_fast_path:
            result.fast_path_count += 1
            result.per_domain_fast[domain] += 1
            if decision_correct:
                result.fast_path_correct += 1
                result.per_domain_correct[domain] += 1
            else:
                # Record error details for per-domain analysis.
                result.fast_path_errors.append(
                    {
                        "index": i,
                        "domain": domain,
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
# Error analysis — matches Exp 145 for comparison
# ---------------------------------------------------------------------------


def analyze_errors(errors: list[dict], mode_name: str) -> dict:
    """Analyze fast-path errors by domain, JEPA probability, and response length.

    **Detailed explanation for engineers:**
        Fast-path errors = JEPA predicted "low risk" but full pipeline would
        have found a violation. We analyze by domain, average JEPA max-prob
        at error, and whether short responses (≤50 tokens) are overrepresented.

        v2 improvement hypothesis: code errors should decrease (v2 code AUROC
        0.776 vs v1 0.706); logic may stay similar or worsen (v2 logic AUROC
        0.479 vs v1 0.534 — v2 regressed slightly on logic per Exp 155).

    Args:
        errors: List of error dicts from ModeResult.fast_path_errors.
        mode_name: Label for the mode being analyzed.

    Returns:
        Dict with n_errors, domain_counts, avg_jepa_max_prob,
        short_response_fraction, most_common_domain.
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
    """Run Experiment 156: JEPA fast-path v2 validation benchmark."""
    t_start = time.perf_counter()
    rng = random.Random(SEED)

    print("=" * 70)
    print("Experiment 156: JEPA Fast-Path v2 Validation Benchmark")
    print("=" * 70)

    # --- Load predictors ---
    # v2 is required (main subject of this experiment).
    if not V2_PREDICTOR_PATH.exists():
        print(f"ERROR: v2 predictor not found at {V2_PREDICTOR_PATH}")
        print("Run experiment_155_retrain_jepa_v2.py first.")
        sys.exit(1)

    print(f"\nLoading JEPA v2 predictor from {V2_PREDICTOR_PATH}...")
    predictor_v2 = JEPAViolationPredictor()
    predictor_v2.load(str(V2_PREDICTOR_PATH))
    print("  v2 predictor loaded.")

    # v1 is loaded for direct per-threshold comparison (optional — skip if missing).
    predictor_v1: JEPAViolationPredictor | None = None
    if V1_PREDICTOR_PATH.exists():
        print(f"Loading JEPA v1 predictor from {V1_PREDICTOR_PATH}...")
        predictor_v1 = JEPAViolationPredictor()
        predictor_v1.load(str(V1_PREDICTOR_PATH))
        print("  v1 predictor loaded (for direct comparison).")
    else:
        print(f"  v1 predictor not found at {V1_PREDICTOR_PATH}; skipping v1 comparison.")

    # --- Load Exp 145 results for v1 reference ---
    v1_results: dict = {}
    if V1_RESULTS_PATH.exists():
        with open(V1_RESULTS_PATH) as f:
            v1_results = json.load(f)
        print(f"\nLoaded Exp 145 v1 baseline from {V1_RESULTS_PATH}")
        for mode in v1_results.get("modes", []):
            print(
                f"  v1 {mode['name']}: fast_path={100*mode['fast_path_rate']:.1f}%"
                f"  degradation={100*mode['accuracy_degradation']:.2f}%"
            )

    # --- Generate synthetic Q&A pairs (same seed as Exp 145) ---
    print(f"\nGenerating {N_TOTAL} synthetic Q&A pairs (seed={SEED}, matches Exp 145)...")
    arithmetic_pairs = generate_arithmetic_qa(rng)
    code_pairs = generate_code_qa(rng)
    logic_pairs = generate_logic_qa(rng)
    all_pairs = arithmetic_pairs + code_pairs + logic_pairs

    n_correct = sum(1 for p in all_pairs if p["ground_truth_correct"])
    print(
        f"  {N_ARITHMETIC} arithmetic, {N_CODE} code, {N_LOGIC} logic  "
        f"({n_correct}/{N_TOTAL} = {100*n_correct/N_TOTAL:.1f}% ground-truth correct)"
    )

    # --- Set up pipeline ---
    pipeline = VerifyRepairPipeline(timeout_seconds=10.0)

    # =========================================================================
    # Mode 1: Baseline (no JEPA gating) — establish reference decisions
    # =========================================================================
    print("\n--- Mode: Baseline (no JEPA gating) ---")
    baseline_result = run_benchmark(
        qa_pairs=all_pairs,
        pipeline=pipeline,
        predictor=None,
        threshold=0.5,  # ignored in baseline mode
        mode_name="baseline",
        baseline_decisions=None,
    )

    # Re-run baseline to collect per-question verified decisions for comparison.
    # (run_benchmark above records the result but the decisions are in its loop;
    # we need them as a list for subsequent gating modes.)
    baseline_decisions: list[bool] = []
    for qa in all_pairs:
        vr = pipeline.verify(
            question=qa["question"],
            response=qa["response"],
            domain=qa["domain"],
        )
        baseline_decisions.append(vr.verified)

    n_baseline_verified = sum(baseline_decisions)
    baseline_gt_match = sum(
        1 for i, qa in enumerate(all_pairs)
        if (baseline_decisions[i] == qa["ground_truth_correct"])
    )
    print(f"  Total time: {baseline_result.total_wall_time_s:.3f}s")
    print(f"  Verified: {n_baseline_verified}/{N_TOTAL} = {100*n_baseline_verified/N_TOTAL:.1f}%")
    print(f"  Accuracy vs ground truth: {100*baseline_gt_match/N_TOTAL:.1f}%")

    # =========================================================================
    # Mode 2: v2 predictor at thresholds 0.3, 0.5, 0.7
    # =========================================================================
    v2_results_by_threshold: list[ModeResult] = []
    v1_results_by_threshold: list[ModeResult] = []

    for threshold in THRESHOLDS:
        # --- v2 ---
        mode_name_v2 = f"v2_threshold={threshold}"
        print(f"\n--- Mode: JEPA v2 gate at threshold={threshold} ---")
        v2_mode = run_benchmark(
            qa_pairs=all_pairs,
            pipeline=pipeline,
            predictor=predictor_v2,
            threshold=threshold,
            mode_name=mode_name_v2,
            baseline_decisions=baseline_decisions,
        )
        v2_results_by_threshold.append(v2_mode)

        degradation_v2 = 1.0 - v2_mode.overall_accuracy
        speedup_v2 = baseline_result.total_wall_time_s / max(v2_mode.total_wall_time_s, 1e-9)
        fp_ok = v2_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        deg_ok = degradation_v2 <= TARGET_MAX_DEGRADATION
        status = "PASS" if (fp_ok and deg_ok) else "MISS"

        print(f"  Time: {v2_mode.total_wall_time_s:.3f}s  (speedup: {speedup_v2:.2f}×)")
        print(f"  Fast-path rate: {100*v2_mode.fast_path_rate:.1f}%  (target ≥ {100*TARGET_FASTPATH_RATE:.0f}%)")
        print(f"  Fast-path accuracy: {100*v2_mode.fast_path_accuracy:.1f}%")
        print(f"  Slow-path accuracy: {100*v2_mode.slow_path_accuracy:.1f}%")
        print(f"  Overall accuracy: {100*v2_mode.overall_accuracy:.1f}%")
        print(f"  Accuracy degradation: {100*degradation_v2:.2f}%  (target < {100*TARGET_MAX_DEGRADATION:.0f}%)")
        print(f"  Target [{TARGET_FASTPATH_RATE*100:.0f}% fast-path, <{TARGET_MAX_DEGRADATION*100:.0f}% degradation]: {status}")

        # Per-domain fast-path accuracy.
        domain_acc = v2_mode.per_domain_fast_path_accuracy()
        print(f"  Per-domain fast-path accuracy:")
        for domain, acc in domain_acc.items():
            n_fast = v2_mode.per_domain_fast[domain]
            if n_fast > 0:
                print(f"    {domain:12s}: {100*acc:.1f}%  ({n_fast} fast-path questions)")
            else:
                print(f"    {domain:12s}: — (no fast-path questions)")

        # --- v1 comparison at same threshold (if available) ---
        if predictor_v1 is not None:
            mode_name_v1 = f"v1_threshold={threshold}"
            print(f"\n  [Comparison] JEPA v1 gate at threshold={threshold}...")
            v1_mode = run_benchmark(
                qa_pairs=all_pairs,
                pipeline=pipeline,
                predictor=predictor_v1,
                threshold=threshold,
                mode_name=mode_name_v1,
                baseline_decisions=baseline_decisions,
            )
            v1_results_by_threshold.append(v1_mode)
            degradation_v1 = 1.0 - v1_mode.overall_accuracy
            print(
                f"  v1: fast_path={100*v1_mode.fast_path_rate:.1f}%  "
                f"degradation={100*degradation_v1:.2f}%  "
                f"accuracy={100*v1_mode.overall_accuracy:.1f}%"
            )
            # v2 delta.
            fp_delta = v2_mode.fast_path_rate - v1_mode.fast_path_rate
            deg_delta = degradation_v2 - degradation_v1
            print(
                f"  v2 vs v1 delta: fast_path Δ={100*fp_delta:+.1f}%  "
                f"degradation Δ={100*deg_delta:+.2f}%"
            )

    # =========================================================================
    # Find optimal threshold: highest fast_path_rate that keeps degradation < 2%
    # =========================================================================
    print("\n--- Optimal threshold search ---")
    optimal_threshold: float | None = None
    optimal_mode: ModeResult | None = None

    for mode in v2_results_by_threshold:
        degradation = 1.0 - mode.overall_accuracy
        fp_rate = mode.fast_path_rate
        # Primary criterion: degradation < 2%.
        # Secondary: maximise fast_path_rate among passing thresholds.
        if degradation <= TARGET_MAX_DEGRADATION:
            if optimal_mode is None or fp_rate > optimal_mode.fast_path_rate:
                optimal_threshold = float(mode.name.split("=")[1])
                optimal_mode = mode

    if optimal_mode is not None:
        print(
            f"  Optimal threshold: {optimal_threshold}  "
            f"(fast_path={100*optimal_mode.fast_path_rate:.1f}%  "
            f"degradation={100*(1.0 - optimal_mode.overall_accuracy):.2f}%)"
        )
        target_met = optimal_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        print(f"  Fast-path target (≥{100*TARGET_FASTPATH_RATE:.0f}%): {'MET' if target_met else 'NOT MET'}")
    else:
        print("  No threshold achieved <2% degradation. Target NOT MET.")
        target_met = False

    # =========================================================================
    # Error analysis
    # =========================================================================
    print("\n--- Error analysis (fast-path misses, v2) ---")
    error_analyses: list[dict] = []
    for mode in v2_results_by_threshold:
        analysis = analyze_errors(mode.fast_path_errors, mode.name)
        error_analyses.append(analysis)
        print(f"\n  Mode: {mode.name}")
        print(f"    Total fast-path errors: {analysis['n_errors']}")
        if analysis["n_errors"] > 0:
            print(f"    Errors by domain: {analysis['domain_counts']}")
            print(f"    Most common error domain: {analysis.get('most_common_domain', 'N/A')}")
            if analysis["avg_jepa_max_prob"] is not None:
                print(f"    Avg JEPA max_prob at error: {analysis['avg_jepa_max_prob']:.3f}")
            # v2 hypothesis: code errors should drop (higher code AUROC in v2).
            code_errors = analysis["domain_counts"].get("code", 0)
            total_errors = analysis["n_errors"]
            if total_errors > 0:
                code_frac = code_errors / total_errors
                print(f"    Code error fraction: {100*code_frac:.1f}%  (v1 Exp145: code dominated at threshold=0.3)")

    # =========================================================================
    # Wall-clock summary
    # =========================================================================
    print("\n--- Wall-clock summary ---")
    print(f"  Baseline: {baseline_result.total_wall_time_s:.3f}s ({N_TOTAL} questions, full verify)")
    for mode in v2_results_by_threshold:
        speedup = baseline_result.total_wall_time_s / max(mode.total_wall_time_s, 1e-9)
        print(
            f"  {mode.name}: {mode.total_wall_time_s:.3f}s  "
            f"(speedup {speedup:.2f}×, {100*mode.fast_path_rate:.1f}% fast-path)"
        )
    if v1_results_by_threshold:
        print("\n  v1 comparison (same questions):")
        for mode in v1_results_by_threshold:
            speedup = baseline_result.total_wall_time_s / max(mode.total_wall_time_s, 1e-9)
            print(
                f"  {mode.name}: {mode.total_wall_time_s:.3f}s  "
                f"(speedup {speedup:.2f}×, {100*mode.fast_path_rate:.1f}% fast-path)"
            )

    # =========================================================================
    # Save results
    # =========================================================================
    # Build per-threshold v1 lookup from Exp 145 JSON (fallback if v1 not run live).
    v1_exp145_by_threshold: dict = {}
    for mode in v1_results.get("modes", []):
        v1_exp145_by_threshold[mode["threshold"]] = mode

    thresholds_output: list[dict] = []
    for i, (threshold, v2_mode) in enumerate(zip(THRESHOLDS, v2_results_by_threshold)):
        degradation_v2 = 1.0 - v2_mode.overall_accuracy
        speedup_v2 = baseline_result.total_wall_time_s / max(v2_mode.total_wall_time_s, 1e-9)

        # v1 live run (if available), else fall back to Exp 145 JSON numbers.
        v1_live = v1_results_by_threshold[i] if i < len(v1_results_by_threshold) else None
        v1_json = v1_exp145_by_threshold.get(threshold, {})

        v1_fast_path_rate = v1_live.fast_path_rate if v1_live else v1_json.get("fast_path_rate")
        v1_degradation = (1.0 - v1_live.overall_accuracy) if v1_live else v1_json.get("accuracy_degradation")

        domain_acc = v2_mode.per_domain_fast_path_accuracy()
        domain_counts_v2 = dict(v2_mode.per_domain_fast)

        thresholds_output.append(
            {
                "threshold": threshold,
                "v2": {
                    "fast_path_rate": v2_mode.fast_path_rate,
                    "fast_path_count": v2_mode.fast_path_count,
                    "slow_path_count": v2_mode.slow_path_count,
                    "fast_path_accuracy": v2_mode.fast_path_accuracy,
                    "slow_path_accuracy": v2_mode.slow_path_accuracy,
                    "overall_accuracy": v2_mode.overall_accuracy,
                    "accuracy_degradation": degradation_v2,
                    "n_fast_path_errors": len(v2_mode.fast_path_errors),
                    "total_time_s": v2_mode.total_wall_time_s,
                    "speedup_vs_baseline": speedup_v2,
                    "target_fast_path_met": v2_mode.fast_path_rate >= TARGET_FASTPATH_RATE,
                    "target_degradation_met": degradation_v2 <= TARGET_MAX_DEGRADATION,
                    "per_domain_fast_path_accuracy": {
                        domain: acc for domain, acc in domain_acc.items()
                    },
                    "per_domain_fast_path_count": domain_counts_v2,
                },
                "v1_comparison": {
                    "fast_path_rate": v1_fast_path_rate,
                    "accuracy_degradation": v1_degradation,
                    "fast_path_rate_delta": (
                        v2_mode.fast_path_rate - v1_fast_path_rate
                        if v1_fast_path_rate is not None else None
                    ),
                    "degradation_delta": (
                        degradation_v2 - v1_degradation
                        if v1_degradation is not None else None
                    ),
                    "source": "live_run" if v1_live else "exp_145_json",
                },
            }
        )

    # Per-domain v1 fast-path accuracy at threshold=0.5 from Exp 145 error analysis.
    # v1 had 42 code errors at threshold=0.3 (dominant), 57 arithmetic errors at 0.5.
    v1_domain_analysis = {
        mode["mode"]: mode for mode in v1_results.get("error_analysis", [])
    }

    results: dict = {
        "experiment": 156,
        "description": "JEPA fast-path v2 validation — validates Exp 155 improved predictor",
        "references": {
            "exp_145": "v1 baseline benchmark",
            "exp_155": "v2 predictor training (multi-domain)",
        },
        "predictor_versions": {
            "v1_path": str(V1_PREDICTOR_PATH),
            "v2_path": str(V2_PREDICTOR_PATH),
            "v2_macro_auroc_from_exp155": 0.6588828939009381,
            "v2_code_auroc_from_exp155": 0.7764705882352941,
            "v1_macro_auroc_from_exp155": 0.647760691766653,
        },
        "n_total": N_TOTAL,
        "n_arithmetic": N_ARITHMETIC,
        "n_code": N_CODE,
        "n_logic": N_LOGIC,
        "seed": SEED,
        "targets": {
            "fast_path_rate_min": TARGET_FASTPATH_RATE,
            "max_accuracy_degradation": TARGET_MAX_DEGRADATION,
        },
        "baseline": {
            "total_time_s": baseline_result.total_wall_time_s,
            "n_verified": n_baseline_verified,
            "gt_accuracy": baseline_gt_match / N_TOTAL,
        },
        "thresholds": thresholds_output,
        "optimal_threshold": optimal_threshold,
        "target_met": target_met and (
            optimal_mode is not None and
            optimal_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        ),
        "error_analysis": error_analyses,
        "v1_exp145_reference": {
            "threshold_0_3": v1_exp145_by_threshold.get(0.3, {}),
            "threshold_0_5": v1_exp145_by_threshold.get(0.5, {}),
        },
        "wall_clock_summary": {
            "baseline_s": baseline_result.total_wall_time_s,
            "v2_by_threshold": {
                str(t): mode.total_wall_time_s
                for t, mode in zip(THRESHOLDS, v2_results_by_threshold)
            },
        },
        "total_experiment_time_s": time.perf_counter() - t_start,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Exp 156 — JEPA v2 Fast-Path Validation")
    print("=" * 70)
    print(f"{'Mode':<25}  {'Fast-path':>10}  {'Degradation':>12}  {'Target':>6}")
    print("-" * 60)

    for t, mode in zip(THRESHOLDS, v2_results_by_threshold):
        degradation = 1.0 - mode.overall_accuracy
        fp_ok = mode.fast_path_rate >= TARGET_FASTPATH_RATE
        deg_ok = degradation <= TARGET_MAX_DEGRADATION
        status = "PASS" if (fp_ok and deg_ok) else "MISS"
        print(
            f"  v2 threshold={t:<5}       {100*mode.fast_path_rate:>8.1f}%  "
            f"{100*degradation:>10.2f}%  {status:>6}"
        )

    if v1_exp145_by_threshold:
        print("\n  v1 reference (Exp 145):")
        for t in [0.3, 0.5]:
            m = v1_exp145_by_threshold.get(t, {})
            if m:
                print(
                    f"  v1 threshold={t:<5}       "
                    f"{100*m.get('fast_path_rate', 0):>8.1f}%  "
                    f"{100*m.get('accuracy_degradation', 0):>10.2f}%"
                )

    print("-" * 60)
    if optimal_threshold is not None:
        print(f"  Optimal threshold: {optimal_threshold}")
        if target_met:
            print(f"  OVERALL: TARGET MET ✓ (≥{100*TARGET_FASTPATH_RATE:.0f}% fast-path, <{100*TARGET_MAX_DEGRADATION:.0f}% degradation)")
        else:
            print(f"  OVERALL: Degradation target met but fast-path rate < {100*TARGET_FASTPATH_RATE:.0f}%")
    else:
        print("  OVERALL: TARGET NOT MET (no threshold achieved <2% degradation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
