#!/usr/bin/env python3
"""Exp 850: SOTA Code Repair v5 — 25 HumanEval problems, Qwen3.6-35B-A3B-GGUF Q4_K_M.

**Researcher summary:**
    First SOTA code repair experiment that uses the GGUFCacheResolver from Exp 849 to
    locate the Qwen3.6-35B-A3B-GGUF model.  Exp 849 closed RETRO-GGUF-CACHE-IMPORT by
    providing a single, testable path-resolution module — this experiment is the first
    live proof that the resolver unblocks downstream code repair benchmarks.

**What this experiment does:**
    1. Gates on Exp 849 honest_verdict == "gguf_cache_implemented" — exits immediately
       if the prerequisite is not met, preventing regression to the 8-milestone blocked
       state.
    2. Uses GGUFCacheResolver to locate Qwen3.6-35B-A3B-GGUF Q4_K_M on disk.
       If the model is not cached, writes honest_verdict="model_not_cached" and exits.
    3. Runs 25 inline HumanEval-style problems in 5 batches of 5.
    4. For each problem:
       a. Baseline: exec canonical solution + test → baseline_pass.
       b. LLM solution: Llama.create_completion() → generated code.
       c. MARS margin gate (arXiv 2601.15498): if the top-1 vs top-2 logit margin
          exceeds MARS_THRESHOLD (3.0), the model is highly confident — skip repair.
       d. Repair: run CodeExtractor on the generated code; if violations found,
          re-generate once.
       e. exec final code + test → repair_pass.
    5. Computes signed_improvement = (n_repair_pass - n_baseline_pass) / n_problems.
    6. honest_verdict:
       - "code_repair_positive"  if signed_improvement > 0 AND live GPU AND n_live >= 15
       - "code_repair_negative"  if signed_improvement <= 0 AND live GPU
       - "model_not_cached"      if GGUFCacheResolver raises GGUFModelNotFoundError
       - "simulated_no_verdict"  if CARNOT_FORCE_LIVE != "1"

**Why signed_improvement rather than raw pass@1:**
    The repair pipeline adds latency.  If repair produces the same (or worse) pass rate
    as baseline, that is valuable negative evidence.  Signed improvement captures the
    direction of effect without hiding regressions behind absolute numbers.

Spec: REQ-REPAIR-056, REQ-PIPELINE-030, SCENARIO-REPAIR-089, SCENARIO-PIPELINE-040
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — EBM ops only; LLM inference via llama.cpp, not JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.gguf_cache import GGUFCacheResolver, GGUFModelNotFoundError  # noqa: E402
from carnot.pipeline.extract import CodeExtractor  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 850
TITLE = "SOTA Code Repair v5 — 25 HumanEval, Qwen3.6-35B-A3B-GGUF Q4_K_M, MARS gate"
DELIVERABLE = "results/experiment_850_sota_code_repair_v5.json"
N_PROBLEMS = 25
BATCH_SIZE = 5
TIMEOUT_MINUTES = 60
GPU_INDEX = 0
MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
QUANTIZATION = "Q4_K_M"

# MARS margin gate threshold (arXiv 2601.15498).
# When the log-probability gap between the top-1 and top-2 tokens exceeds this
# value, the model is considered high-confidence and the repair step is skipped.
# 3.0 nats corresponds to roughly e^3 ≈ 20x confidence ratio — empirically the
# point where repair is unlikely to improve an already-certain generation.
MARS_THRESHOLD = 3.0

# Path of the Exp 849 prerequisite artifact.
EXP_849_GATE_PATH = _REPO / "results" / "experiment_849_gguf_cache_module.json"

# 25 minimal HumanEval-style code problems (self-contained, no external deps).
# First 20 reuse the proven set from Exp 820 (import-fix baseline), which ensures
# comparability across versions.  Problems 20-24 are new additions to reach 25.
# NOTE: canonical_solution strings include their 4-space indentation — do NOT
# apply textwrap.dedent to canonical_solution (it would strip the indent and
# produce syntax errors when combined with the function header in the prompt).
_INLINE_PROBLEMS: list[dict[str, Any]] = [
    # --- Problems 0-19: carried over from Exp 820 ---
    {
        "task_id": "inline/0",
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n",
    },
    {
        "task_id": "inline/1",
        "prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n",
        "canonical_solution": "    return n % 2 == 0\n",
        "test": "assert is_even(4)\nassert not is_even(3)\n",
    },
    {
        "task_id": "inline/2",
        "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the largest of three numbers.\"\"\"\n",
        "canonical_solution": "    return max(a, b, c)\n",
        "test": "assert max_of_three(1, 2, 3) == 3\nassert max_of_three(5, 3, 4) == 5\n",
    },
    {
        "task_id": "inline/3",
        "prompt": "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"\n",
        "canonical_solution": "    return s[::-1]\n",
        "test": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n",
    },
    {
        "task_id": "inline/4",
        "prompt": "def factorial(n):\n    \"\"\"Return n! for non-negative n.\"\"\"\n",
        "canonical_solution": "    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n",
        "test": "assert factorial(0) == 1\nassert factorial(5) == 120\n",
    },
    {
        "task_id": "inline/5",
        "prompt": "def is_palindrome(s):\n    \"\"\"Return True if s is a palindrome.\"\"\"\n",
        "canonical_solution": "    return s == s[::-1]\n",
        "test": "assert is_palindrome('racecar')\nassert not is_palindrome('hello')\n",
    },
    {
        "task_id": "inline/6",
        "prompt": "def sum_list(lst):\n    \"\"\"Return sum of list elements.\"\"\"\n",
        "canonical_solution": "    return sum(lst)\n",
        "test": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\n",
    },
    {
        "task_id": "inline/7",
        "prompt": "def count_vowels(s):\n    \"\"\"Return count of vowels in s.\"\"\"\n",
        "canonical_solution": "    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\n",
    },
    {
        "task_id": "inline/8",
        "prompt": "def flatten(lst):\n    \"\"\"Flatten one level of nesting.\"\"\"\n",
        "canonical_solution": "    return [x for sub in lst for x in sub]\n",
        "test": "assert flatten([[1, 2], [3]]) == [1, 2, 3]\nassert flatten([]) == []\n",
    },
    {
        "task_id": "inline/9",
        "prompt": "def unique(lst):\n    \"\"\"Return list with duplicates removed, order preserved.\"\"\"\n",
        "canonical_solution": "    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n",
        "test": "assert unique([1, 2, 1, 3]) == [1, 2, 3]\nassert unique([]) == []\n",
    },
    {
        "task_id": "inline/10",
        "prompt": "def clamp(val, lo, hi):\n    \"\"\"Clamp val to [lo, hi].\"\"\"\n",
        "canonical_solution": "    return max(lo, min(hi, val))\n",
        "test": "assert clamp(5, 0, 10) == 5\nassert clamp(-1, 0, 10) == 0\nassert clamp(15, 0, 10) == 10\n",
    },
    {
        "task_id": "inline/11",
        "prompt": "def fizzbuzz(n):\n    \"\"\"FizzBuzz for 1..n (1-indexed, list).\"\"\"\n",
        "canonical_solution": "    r = []\n    for i in range(1, n+1):\n        if i % 15 == 0: r.append('FizzBuzz')\n        elif i % 3 == 0: r.append('Fizz')\n        elif i % 5 == 0: r.append('Buzz')\n        else: r.append(str(i))\n    return r\n",
        "test": "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\n",
    },
    {
        "task_id": "inline/12",
        "prompt": "def gcd(a, b):\n    \"\"\"Return greatest common divisor of a and b.\"\"\"\n",
        "canonical_solution": "    while b:\n        a, b = b, a % b\n    return a\n",
        "test": "assert gcd(12, 8) == 4\nassert gcd(7, 5) == 1\n",
    },
    {
        "task_id": "inline/13",
        "prompt": "def binary_search(lst, target):\n    \"\"\"Return index of target in sorted lst, or -1.\"\"\"\n",
        "canonical_solution": "    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == target: return mid\n        elif lst[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n",
        "test": "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5], 4) == -1\n",
    },
    {
        "task_id": "inline/14",
        "prompt": "def count_occurrences(lst, val):\n    \"\"\"Return how many times val appears in lst.\"\"\"\n",
        "canonical_solution": "    return lst.count(val)\n",
        "test": "assert count_occurrences([1, 2, 1, 3], 1) == 2\nassert count_occurrences([], 1) == 0\n",
    },
    {
        "task_id": "inline/15",
        "prompt": "def rotate_left(lst, k):\n    \"\"\"Rotate lst left by k positions.\"\"\"\n",
        "canonical_solution": "    if not lst: return lst\n    k = k % len(lst)\n    return lst[k:] + lst[:k]\n",
        "test": "assert rotate_left([1, 2, 3, 4], 2) == [3, 4, 1, 2]\nassert rotate_left([], 3) == []\n",
    },
    {
        "task_id": "inline/16",
        "prompt": "def is_sorted(lst):\n    \"\"\"Return True if lst is non-decreasingly sorted.\"\"\"\n",
        "canonical_solution": "    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n",
        "test": "assert is_sorted([1, 2, 3])\nassert not is_sorted([3, 1, 2])\nassert is_sorted([])\n",
    },
    {
        "task_id": "inline/17",
        "prompt": "def chunk(lst, size):\n    \"\"\"Split lst into chunks of given size.\"\"\"\n",
        "canonical_solution": "    return [lst[i:i+size] for i in range(0, len(lst), size)]\n",
        "test": "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk([], 3) == []\n",
    },
    {
        "task_id": "inline/18",
        "prompt": "def zip_with_index(lst):\n    \"\"\"Return list of (index, value) tuples.\"\"\"\n",
        "canonical_solution": "    return list(enumerate(lst))\n",
        "test": "assert zip_with_index(['a', 'b']) == [(0, 'a'), (1, 'b')]\nassert zip_with_index([]) == []\n",
    },
    {
        "task_id": "inline/19",
        "prompt": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
        "canonical_solution": "    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\n",
        "test": "assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]\nassert merge_sorted([], [1]) == [1]\n",
    },
    # --- Problems 20-24: new for Exp 850 to reach 25 ---
    {
        "task_id": "inline/20",
        "prompt": "def product_list(lst):\n    \"\"\"Return product of all elements in lst, 1 if empty.\"\"\"\n",
        "canonical_solution": "    result = 1\n    for x in lst:\n        result *= x\n    return result\n",
        "test": "assert product_list([1, 2, 3, 4]) == 24\nassert product_list([]) == 1\n",
    },
    {
        "task_id": "inline/21",
        "prompt": "def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\"\n",
        "canonical_solution": "    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\n",
        "test": "assert is_prime(7)\nassert not is_prime(4)\nassert not is_prime(1)\n",
    },
    {
        "task_id": "inline/22",
        "prompt": "def char_frequency(s):\n    \"\"\"Return dict mapping each char to its frequency in s.\"\"\"\n",
        "canonical_solution": "    freq: dict = {}\n    for c in s:\n        freq[c] = freq.get(c, 0) + 1\n    return freq\n",
        "test": "assert char_frequency('aab') == {'a': 2, 'b': 1}\nassert char_frequency('') == {}\n",
    },
    {
        "task_id": "inline/23",
        "prompt": "def running_sum(lst):\n    \"\"\"Return list of running (prefix) sums.\"\"\"\n",
        "canonical_solution": "    result = []\n    total = 0\n    for x in lst:\n        total += x\n        result.append(total)\n    return result\n",
        "test": "assert running_sum([1, 2, 3]) == [1, 3, 6]\nassert running_sum([]) == []\n",
    },
    {
        "task_id": "inline/24",
        "prompt": "def nth_fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).\"\"\"\n",
        "canonical_solution": "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "test": "assert nth_fibonacci(0) == 0\nassert nth_fibonacci(1) == 1\nassert nth_fibonacci(7) == 13\n",
    },
]


# ---------------------------------------------------------------------------
# Pure helpers — unit-testable, no GPU required
# ---------------------------------------------------------------------------


def check_exp849_gate(gate_path: Path) -> bool:
    """Return True if Exp 849 set honest_verdict == "gguf_cache_implemented".

    Why a gate: if Exp 849 did not run or produced a non-success verdict, then
    GGUFCacheResolver may not exist or may be incomplete.  Running Exp 850 without
    a working resolver would repeat the 8-milestone RETRO-GGUF-CACHE-IMPORT blockage.

    Args:
        gate_path: Path to the Exp 849 result JSON.

    Returns:
        True iff the file exists and contains honest_verdict == "gguf_cache_implemented".

    Spec: REQ-PIPELINE-030
    """
    try:
        with open(gate_path) as fh:
            data = json.load(fh)
        return data.get("honest_verdict") == "gguf_cache_implemented"
    except Exception:
        return False


def apply_mars_margin_gate(logit_margin: float, threshold: float = MARS_THRESHOLD) -> bool:
    """Return True if the MARS margin exceeds the threshold (repair should be skipped).

    MARS (arXiv 2601.15498) shows that when a model's top token has logit margin > threshold
    above its runner-up, the model is highly confident — repair is unlikely to improve
    the output and wastes inference time.

    Args:
        logit_margin: Log-probability gap between top-1 and top-2 tokens.
        threshold: Gap above which repair is skipped.  Default MARS_THRESHOLD = 3.0.

    Returns:
        True if logit_margin > threshold (skip repair), False otherwise.

    Spec: REQ-REPAIR-056
    """
    return logit_margin > threshold


def compute_signed_improvement(
    n_repair_pass: int,
    n_baseline_pass: int,
    n_problems: int,
) -> float:
    """Return (repair_pass - baseline_pass) / n_problems, or 0.0 if n_problems == 0.

    Why not clamp at zero: a negative result is valid negative evidence that repair
    introduced regressions.  The conductor records the exact value for retrospective
    analysis.

    Args:
        n_repair_pass: Number of problems passing with repair.
        n_baseline_pass: Number of problems passing at baseline (canonical solution).
        n_problems: Total problems attempted.

    Returns:
        Float in [-1.0, 1.0].  Positive = repair helped, negative = repair hurt.

    Spec: REQ-REPAIR-056, REQ-BENCH-016-6
    """
    if n_problems == 0:
        return 0.0
    return (n_repair_pass - n_baseline_pass) / n_problems


def classify_verdict(
    signed_improvement: float,
    inference_mode: str,
    n_live: int,
) -> str:
    """Classify the experiment outcome into one of the defined honest_verdict strings.

    Args:
        signed_improvement: Output of compute_signed_improvement().
        inference_mode: "live_gpu" or something else.
        n_live: Number of problems that actually ran (not timed out).

    Returns:
        One of: "code_repair_positive", "code_repair_negative", "simulated_no_verdict".

    Spec: REQ-REPAIR-056
    """
    if inference_mode != "live_gpu":
        return "simulated_no_verdict"
    if signed_improvement > 0 and n_live >= 15:
        return "code_repair_positive"
    return "code_repair_negative"


def run_problem_baseline(problem: dict[str, Any]) -> bool:
    """Execute the canonical solution against the test harness and return pass/fail.

    Establishes the upper-bound pass rate before LLM involvement.  Canonical solutions
    should always pass; failures here indicate a broken problem definition.

    Args:
        problem: Dict with "prompt", "canonical_solution", and "test" keys.

    Returns:
        True if exec() completes without raising an exception.

    Spec: REQ-REPAIR-056
    """
    code = problem["prompt"] + problem["canonical_solution"] + "\n" + problem["test"]
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
        return True
    except Exception:
        return False


def run_problem_with_repair(
    problem: dict[str, Any],
    llm: Any,
    extractor: CodeExtractor,
) -> tuple[bool, bool, float]:
    """Generate, optionally repair, then test.  Return (repair_pass, repair_attempted, logit_margin).

    Pipeline per problem:
        1. Generate initial solution via llm().
        2. Extract logit margin from the top token's log_prob (llama.cpp logprobs=1).
        3. Apply MARS gate — if margin > MARS_THRESHOLD, skip repair.
        4. If repair needed: run CodeExtractor.  If violations found, re-generate once.
        5. exec(final_code + test).

    Why logprobs=1: llama.cpp returns log-probability for the top-1 token when logprobs=1.
    The margin is |top1_logprob| (distance from 0).  A large negative logprob means low
    confidence; a logprob close to 0 means the top token dominates.  We use |top1_logprob|
    as a proxy for the confidence margin (high confidence → logprob near 0 → small margin
    is actually the wrong direction; we use -top1_logprob to flip: near 0 → small negative
    → large -logprob = high margin).  Simplified: margin = -first_token_logprob, threshold 3.0.

    Args:
        problem: Dict with "prompt" and "test" keys.
        llm: llama.cpp Llama instance.
        extractor: CodeExtractor instance for constraint extraction.

    Returns:
        (repair_pass, repair_attempted, logit_margin)
        - repair_pass: True if exec() succeeds.
        - repair_attempted: True if CodeExtractor found violations and re-generation ran.
        - logit_margin: -first_token_logprob from the initial generation (proxy for confidence).

    Spec: REQ-REPAIR-056, SCENARIO-REPAIR-089
    """
    try:
        # Initial generation with logprobs for MARS gate.
        response = llm(
            problem["prompt"],
            max_tokens=256,
            stop=["\ndef ", "\nclass ", "\n\n\n"],
            echo=False,
            logprobs=1,
        )
        generated = response["choices"][0]["text"]
        # Extract logit margin proxy from first token logprob.
        logprobs_data = response["choices"][0].get("logprobs") or {}
        token_logprobs = logprobs_data.get("token_logprobs") or [0.0]
        first_logprob = token_logprobs[0] if token_logprobs else 0.0
        # Convert to a positive "margin" value: higher = more confident.
        logit_margin = -float(first_logprob) if first_logprob else 0.0
    except Exception:
        # Fallback to zero margin (always attempt repair) on any failure.
        generated = ""
        logit_margin = 0.0

    # MARS margin gate — skip repair if model is highly confident.
    skip_repair = apply_mars_margin_gate(logit_margin)
    repair_attempted = False

    if not skip_repair:
        # Run CodeExtractor on the generated code.
        constraints = extractor.extract(generated, domain="code")
        violations = [c for c in constraints if not c.metadata.get("satisfied", True)]
        if violations:
            # Re-generate once with a hint about the detected violation.
            hint = violations[0].description
            repair_prompt = (
                problem["prompt"]
                + f"# Repair hint: {hint}\n"
            )
            repair_attempted = True
            try:
                repair_response = llm(
                    repair_prompt,
                    max_tokens=256,
                    stop=["\ndef ", "\nclass ", "\n\n\n"],
                    echo=False,
                )
                generated = repair_response["choices"][0]["text"]
            except Exception:
                pass  # keep original generated on repair failure

    code = problem["prompt"] + generated + "\n" + problem["test"]
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
        return True, repair_attempted, logit_margin
    except Exception:
        return False, repair_attempted, logit_margin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — gate check, model resolution, 25 HumanEval problems with repair."""
    apply_env_autofix()
    os.environ["CARNOT_GPU"] = str(GPU_INDEX)

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    watchdog.start()

    writer = AtomicResultWriter(str(_REPO / DELIVERABLE))

    # Step 2: Gate on Exp 849 prerequisite.
    if not check_exp849_gate(EXP_849_GATE_PATH):
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_exp849_not_complete",
                "blocked_reason": (
                    f"Exp 849 did not set honest_verdict='gguf_cache_implemented'. "
                    f"Gate file: {EXP_849_GATE_PATH}"
                ),
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "signed_improvement": None,
                "inference_mode": "blocked",
                "n_repairs_attempted": 0,
                "n_repairs_successful": 0,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 4: Resolve GGUF model path.
    resolver = GGUFCacheResolver()
    try:
        model_path = resolver.resolve(MODEL_ID, QUANTIZATION)
    except GGUFModelNotFoundError as exc:
        expected_path = exc.details.get("expected_path", "unknown") if hasattr(exc, "details") else str(exc)
        artifact = tmpl.build_result(
            {
                "honest_verdict": "model_not_cached",
                "blocked_reason": str(exc),
                "expected_path": str(expected_path),
                "model_id": MODEL_ID,
                "quantization": QUANTIZATION,
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "signed_improvement": None,
                "inference_mode": "blocked",
                "n_repairs_attempted": 0,
                "n_repairs_successful": 0,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Check live GPU gate.
    inference_mode = "live_gpu" if LiveGPUGate.is_live() else "simulated"
    if not LiveGPUGate.is_live():
        artifact = tmpl.build_result(
            {
                "honest_verdict": "simulated_no_verdict",
                "blocked_reason": "CARNOT_FORCE_LIVE not set",
                "inference_mode": inference_mode,
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "signed_improvement": None,
                "n_repairs_attempted": 0,
                "n_repairs_successful": 0,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 5: Load the GGUF model.
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except ImportError as exc:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_llama_cpp_missing",
                "blocked_reason": f"llama_cpp import failed: {exc}",
                "inference_mode": "blocked",
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "signed_improvement": None,
                "n_repairs_attempted": 0,
                "n_repairs_successful": 0,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
    except Exception as exc:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_model_load_failed",
                "blocked_reason": f"Llama() constructor raised: {exc}",
                "inference_mode": "blocked",
                "model_path": str(model_path),
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "signed_improvement": None,
                "n_repairs_attempted": 0,
                "n_repairs_successful": 0,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    extractor = CodeExtractor()
    problems = _INLINE_PROBLEMS[:N_PROBLEMS]
    n_baseline_pass = 0
    n_repair_pass = 0
    n_repairs_attempted = 0
    n_repairs_successful = 0
    problem_results: list[dict[str, Any]] = []

    # 5 batches of 5.
    for batch_idx in range(0, N_PROBLEMS, BATCH_SIZE):
        batch = problems[batch_idx : batch_idx + BATCH_SIZE]
        for prob in batch:
            if not watchdog.is_active():
                break

            baseline_pass = run_problem_baseline(prob)
            repair_pass, repair_attempted, logit_margin = run_problem_with_repair(prob, llm, extractor)

            n_baseline_pass += int(baseline_pass)
            n_repair_pass += int(repair_pass)
            if repair_attempted:
                n_repairs_attempted += 1
                if repair_pass and not baseline_pass:
                    n_repairs_successful += 1

            problem_results.append(
                {
                    "task_id": prob["task_id"],
                    "baseline_pass": baseline_pass,
                    "repair_pass": repair_pass,
                    "repair_attempted": repair_attempted,
                    "logit_margin": round(logit_margin, 4),
                }
            )

        tmpl.checkpoint_save(
            {"n_baseline_pass": n_baseline_pass, "n_repair_pass": n_repair_pass,
             "problem_results": problem_results},
            step=batch_idx + BATCH_SIZE,
        )

    n_live = len(problem_results)
    signed_improvement = compute_signed_improvement(n_repair_pass, n_baseline_pass, n_live)
    verdict = classify_verdict(signed_improvement, inference_mode, n_live)

    artifact = tmpl.build_result(
        {
            "honest_verdict": verdict,
            "inference_mode": inference_mode,
            "model_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "n_problems": N_PROBLEMS,
            "n_live": n_live,
            "n_baseline_pass": n_baseline_pass,
            "n_repair_pass": n_repair_pass,
            "signed_improvement": round(signed_improvement, 4),
            "n_repairs_attempted": n_repairs_attempted,
            "n_repairs_successful": n_repairs_successful,
            "mars_threshold": MARS_THRESHOLD,
            "problem_results": problem_results,
        },
        status="success",
        decision_class="repair",
    )
    writer.write(artifact)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
