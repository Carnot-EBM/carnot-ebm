#!/usr/bin/env python3
"""Exp 857: SOTA Code Repair v6 — GGUFCacheResolver download + 25 HumanEval, Qwen3.6-35B-A3B.

**Researcher summary:**
    Extends GGUFCacheResolver with huggingface_hub download capability so the
    model can be fetched automatically when not already cached.  Runs 25
    HumanEval problems with Qwen3.6-35B-A3B-GGUF on GPU 1 via CARNOT_DUAL_GPU
    wiring to measure the signed improvement of Carnot verify-repair over baseline.

**What this experiment does:**
    1. Gate-checks Exp 855 (live_env_fixed=True) and Exp 856 (dual_gpu_deployed=True).
       Exits with status="blocked" if either gate is not met.
    2. Loads session environment via EnvPropagationGuard to pick up CARNOT_FORCE_LIVE
       that was persisted by the Exp 855 preflight.
    3. Asserts CARNOT_FORCE_LIVE=1 is in the process environment (prevents silent
       simulation fallback — the root cause of RETRO-015 recurrence in Exp 853).
    4. Resolves Qwen3.6-35B-A3B-GGUF Q4_K_M via GGUFCacheResolver.  If the model
       is not on disk, the resolver now calls huggingface_hub.hf_hub_download
       automatically (new in this experiment — closes RETRO-SOTA-MODEL-DOWNLOAD).
    5. Runs 25 inline HumanEval-style problems in 5 batches of 5.
    6. For each problem: baseline exec → LLM generation with llama.cpp →
       Carnot VerifyRepairPipeline repair → eval.
    7. Computes signed_improvement, sets honest_verdict.

**honest_verdict meanings:**
    - "positive_repair"       — signed_improvement > 0, confirmed live GPU
    - "live_no_improvement"   — live GPU, signed_improvement <= 0
    - "simulation_fallback"   — CARNOT_FORCE_LIVE not set (env not propagated)
    - "blocked"               — gate failure, model load failure, or assertion error

Spec: REQ-VR-020, SCENARIO-VR-030, REQ-PIPELINE-030, SCENARIO-PIPELINE-040
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

# Force CPU JAX — EBM ops only; LLM inference runs through llama.cpp.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.gguf_cache import GGUFCacheResolver, GGUFCacheConfig, GGUFModelNotFoundError  # noqa: E402
from carnot.pipeline.extract import CodeExtractor  # noqa: E402
from scripts.experiment_template import ExperimentTemplate, EnvPropagationGuard  # noqa: E402

EXP_ID = 857
TITLE = "SOTA Code Repair v6 — GGUFCacheResolver download + 25 HumanEval, Qwen3.6-35B-A3B"
DELIVERABLE = "results/experiment_857_sota_code_repair_v6.json"
N_PROBLEMS = 25
BATCH_SIZE = 5
TIMEOUT_MINUTES = 90
GPU_INDEX = 1
MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
QUANTIZATION = "Q4_K_M"

# Gate artifact paths.
EXP_855_GATE_PATH = _REPO / "results" / "experiment_855_preflight_v15.json"
EXP_856_GATE_PATH = _REPO / "results" / "experiment_856_dualgpu_production.json"


# ---------------------------------------------------------------------------
# 25 inline HumanEval-style problems (carried from Exp 850/853 for comparability).
# ---------------------------------------------------------------------------
_INLINE_PROBLEMS: list[dict[str, Any]] = [
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
        "prompt": "def max_of_two(a, b):\n    \"\"\"Return the larger of a and b.\"\"\"\n",
        "canonical_solution": "    return a if a > b else b\n",
        "test": "assert max_of_two(3, 5) == 5\nassert max_of_two(7, 2) == 7\n",
    },
    {
        "task_id": "inline/3",
        "prompt": "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"\n",
        "canonical_solution": "    return s[::-1]\n",
        "test": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n",
    },
    {
        "task_id": "inline/4",
        "prompt": "def sum_list(lst):\n    \"\"\"Return the sum of all elements in lst.\"\"\"\n",
        "canonical_solution": "    return sum(lst)\n",
        "test": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\n",
    },
    {
        "task_id": "inline/5",
        "prompt": "def factorial(n):\n    \"\"\"Return n! for non-negative integer n.\"\"\"\n",
        "canonical_solution": "    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n",
        "test": "assert factorial(0) == 1\nassert factorial(5) == 120\n",
    },
    {
        "task_id": "inline/6",
        "prompt": "def is_palindrome(s):\n    \"\"\"Return True if s is a palindrome.\"\"\"\n",
        "canonical_solution": "    return s == s[::-1]\n",
        "test": "assert is_palindrome('racecar')\nassert not is_palindrome('hello')\n",
    },
    {
        "task_id": "inline/7",
        "prompt": "def count_vowels(s):\n    \"\"\"Return the number of vowels in s.\"\"\"\n",
        "canonical_solution": "    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\n",
    },
    {
        "task_id": "inline/8",
        "prompt": "def flatten(lst):\n    \"\"\"Flatten a one-level nested list.\"\"\"\n",
        "canonical_solution": "    return [x for sublist in lst for x in sublist]\n",
        "test": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\nassert flatten([]) == []\n",
    },
    {
        "task_id": "inline/9",
        "prompt": "def unique(lst):\n    \"\"\"Return sorted list of unique elements.\"\"\"\n",
        "canonical_solution": "    return sorted(set(lst))\n",
        "test": "assert unique([3, 1, 2, 1]) == [1, 2, 3]\nassert unique([]) == []\n",
    },
    {
        "task_id": "inline/10",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number (0-indexed).\"\"\"\n",
        "canonical_solution": "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "test": "assert fibonacci(0) == 0\nassert fibonacci(7) == 13\n",
    },
    {
        "task_id": "inline/11",
        "prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
        "canonical_solution": "    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        "test": "assert is_prime(7)\nassert not is_prime(9)\nassert not is_prime(1)\n",
    },
    {
        "task_id": "inline/12",
        "prompt": "def rotate_list(lst, k):\n    \"\"\"Rotate lst left by k positions.\"\"\"\n",
        "canonical_solution": "    if not lst:\n        return []\n    k = k % len(lst)\n    return lst[k:] + lst[:k]\n",
        "test": "assert rotate_list([1, 2, 3, 4], 1) == [2, 3, 4, 1]\nassert rotate_list([], 3) == []\n",
    },
    {
        "task_id": "inline/13",
        "prompt": "def word_count(text):\n    \"\"\"Return dict mapping each word to its count.\"\"\"\n",
        "canonical_solution": "    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n",
        "test": "assert word_count('a b a') == {'a': 2, 'b': 1}\nassert word_count('') == {}\n",
    },
    {
        "task_id": "inline/14",
        "prompt": "def gcd(a, b):\n    \"\"\"Return the greatest common divisor of a and b.\"\"\"\n",
        "canonical_solution": "    while b:\n        a, b = b, a % b\n    return a\n",
        "test": "assert gcd(12, 8) == 4\nassert gcd(7, 5) == 1\n",
    },
    {
        "task_id": "inline/15",
        "prompt": "def binary_search(lst, target):\n    \"\"\"Return index of target in sorted lst, or -1 if absent.\"\"\"\n",
        "canonical_solution": "    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == target:\n            return mid\n        elif lst[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        "test": "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5], 2) == -1\n",
    },
    {
        "task_id": "inline/16",
        "prompt": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into a sorted list.\"\"\"\n",
        "canonical_solution": "    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\n",
        "test": "assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]\nassert merge_sorted([], [1]) == [1]\n",
    },
    {
        "task_id": "inline/17",
        "prompt": "def caesar_cipher(text, shift):\n    \"\"\"Apply Caesar cipher with the given shift to alphabetic chars.\"\"\"\n",
        "canonical_solution": "    result = []\n    for c in text:\n        if c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            result.append(chr((ord(c) - base + shift) % 26 + base))\n        else:\n            result.append(c)\n    return ''.join(result)\n",
        "test": "assert caesar_cipher('abc', 1) == 'bcd'\nassert caesar_cipher('z', 1) == 'a'\n",
    },
    {
        "task_id": "inline/18",
        "prompt": "def matrix_transpose(matrix):\n    \"\"\"Return the transpose of a 2D list.\"\"\"\n",
        "canonical_solution": "    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix)]\n",
        "test": "assert matrix_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\nassert matrix_transpose([]) == []\n",
    },
    {
        "task_id": "inline/19",
        "prompt": "def running_sum(lst):\n    \"\"\"Return list of running (prefix) sums.\"\"\"\n",
        "canonical_solution": "    result = []\n    total = 0\n    for x in lst:\n        total += x\n        result.append(total)\n    return result\n",
        "test": "assert running_sum([1, 2, 3]) == [1, 3, 6]\nassert running_sum([]) == []\n",
    },
    {
        "task_id": "inline/20",
        "prompt": "def count_words(sentence):\n    \"\"\"Return the number of words in sentence.\"\"\"\n",
        "canonical_solution": "    return len(sentence.split())\n",
        "test": "assert count_words('hello world') == 2\nassert count_words('') == 0\n",
    },
    {
        "task_id": "inline/21",
        "prompt": "def clamp(value, lo, hi):\n    \"\"\"Clamp value to the inclusive range [lo, hi].\"\"\"\n",
        "canonical_solution": "    return max(lo, min(hi, value))\n",
        "test": "assert clamp(5, 1, 10) == 5\nassert clamp(-1, 0, 10) == 0\nassert clamp(15, 0, 10) == 10\n",
    },
    {
        "task_id": "inline/22",
        "prompt": "def zip_to_dict(keys, values):\n    \"\"\"Zip two lists into a dict.\"\"\"\n",
        "canonical_solution": "    return dict(zip(keys, values))\n",
        "test": "assert zip_to_dict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}\nassert zip_to_dict([], []) == {}\n",
    },
    {
        "task_id": "inline/23",
        "prompt": "def deep_copy_list(lst):\n    \"\"\"Return a shallow copy of lst.\"\"\"\n",
        "canonical_solution": "    return lst[:]\n",
        "test": "orig = [1, 2, 3]\ncopy = deep_copy_list(orig)\nassert copy == orig\nassert copy is not orig\n",
    },
    {
        "task_id": "inline/24",
        "prompt": "def chunk_list(lst, size):\n    \"\"\"Split lst into consecutive chunks of at most size elements.\"\"\"\n",
        "canonical_solution": "    return [lst[i:i + size] for i in range(0, len(lst), size)]\n",
        "test": "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_list([], 3) == []\n",
    },
]


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def check_exp855_gate(path: Path = EXP_855_GATE_PATH) -> bool:
    """Return True iff Exp 855 artifact has live_env_fixed=True.

    **Detailed explanation for engineers:**
        Exp 855 fixed the EnvPropagationGuard so that CARNOT_FORCE_LIVE persists
        across subprocess boundaries.  Without that fix, every experiment silently
        falls through to simulated inference.  This gate prevents re-opening
        RETRO-LIVE-ENV-NOT-PROPAGATED.

    Spec: REQ-INFRA-070, SCENARIO-INFRA-080
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
        return bool(data.get("live_env_fixed"))
    except Exception:
        return False


def check_exp856_gate(path: Path = EXP_856_GATE_PATH) -> bool:
    """Return True iff Exp 856 artifact has dual_gpu_deployed=True.

    **Detailed explanation for engineers:**
        Exp 856 wired DualGPURunner into VerifyRepairPipeline and ThreeTierPipeline.
        The code repair v6 experiment uses CARNOT_DUAL_GPU=1, which activates that
        wiring.  If the wiring is not deployed, dual-GPU env var has no effect and
        we cannot claim GPU-parallel repair was tested.

    Spec: REQ-GPU-010, SCENARIO-GPU-020
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
        return bool(data.get("dual_gpu_deployed"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Problem execution helpers
# ---------------------------------------------------------------------------

def run_problem_baseline(problem: dict[str, Any]) -> bool:
    """Execute the canonical solution and return True if the test passes.

    **Detailed explanation for engineers:**
        Constructs a minimal Python module string from the problem's prompt
        (function signature + docstring) and canonical_solution, then execs it
        in a fresh namespace.  The test string is appended and executed in the
        same namespace so that the function is visible to the assertions.

        Why exec() instead of importlib?  The problems are in-memory strings,
        not files.  exec() is the standard approach for dynamic code evaluation
        in HumanEval-style benchmarks (see OpenAI's original eval harness).

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    code = problem["prompt"] + problem["canonical_solution"]
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
        exec(problem["test"], ns)  # noqa: S102
        return True
    except Exception:
        return False


def run_problem_with_repair(
    problem: dict[str, Any],
    llm: Any,
    extractor: "CodeExtractor",
) -> tuple[bool, bool, float]:
    """Generate a solution via LLM, optionally repair, exec and return result.

    **Detailed explanation for engineers:**
        1. Calls llm.create_completion(prompt) with logprobs=1.
        2. Extracts the top-1 vs top-2 logit margin (MARS gate — arXiv 2601.15498).
           If the margin exceeds MARS_THRESHOLD the model is confident — skip repair.
        3. If repair is not skipped: runs CodeExtractor.extract() on the generated
           code; if there are violations, re-generates once.
        4. Execs the final code + test.

    Returns:
        (pass, repair_attempted, logit_margin)

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    MARS_THRESHOLD = 3.0
    prompt = problem["prompt"]
    entry_fn = prompt.split("(")[0].replace("def ", "").strip()

    try:
        resp = llm.create_completion(
            prompt,
            max_tokens=256,
            temperature=0.0,
            logprobs=1,
        )
    except Exception:
        return False, False, 0.0

    gen_text = resp["choices"][0]["text"]
    logprobs_data = resp["choices"][0].get("logprobs") or {}
    top_logprobs_list = logprobs_data.get("top_logprobs") or []

    logit_margin = 0.0
    if top_logprobs_list:
        first_token_logprobs = top_logprobs_list[0] if top_logprobs_list else {}
        vals = sorted(first_token_logprobs.values(), reverse=True)
        if len(vals) >= 2:
            logit_margin = vals[0] - vals[1]

    repair_attempted = False
    final_code = prompt + gen_text

    if logit_margin <= MARS_THRESHOLD:
        repair_attempted = True
        try:
            violations = extractor.extract(gen_text, domain="code")
            if violations:
                repair_resp = llm.create_completion(
                    f"{prompt}# Fix: {violations[0] if violations else ''}\n",
                    max_tokens=256,
                    temperature=0.0,
                )
                final_code = prompt + repair_resp["choices"][0]["text"]
        except Exception:
            pass

    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(final_code, ns)  # noqa: S102
        exec(problem["test"], ns)  # noqa: S102
        return True, repair_attempted, logit_margin
    except Exception:
        return False, repair_attempted, logit_margin


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_signed_improvement(
    n_repair_pass: int,
    n_baseline_pass: int,
    n_problems: int,
) -> float:
    """Return (repair_pass_rate - baseline_pass_rate), or 0.0 if n_problems == 0.

    **Detailed explanation for engineers:**
        Signed improvement captures *direction* of effect.  A positive value
        means Carnot repair improved pass rate.  A negative value is honest
        negative evidence that repair degraded quality (valuable for calibration).
        Zero means no change.

    Spec: REQ-VR-020
    """
    if n_problems == 0:
        return 0.0
    return (n_repair_pass - n_baseline_pass) / n_problems


def classify_verdict(
    signed_improvement: float,
    inference_mode: str,
) -> str:
    """Classify the experiment outcome into one of the defined honest_verdict strings.

    **Detailed explanation for engineers:**
        inference_mode must be "live_gpu" to claim a real result.  Any other
        value is "simulation_fallback" — a known-honest label for runs where
        CARNOT_FORCE_LIVE was not propagated to the subprocess environment.

    Spec: REQ-VR-020, SCENARIO-VR-030
    """
    if inference_mode != "live_gpu":
        return "simulation_fallback"
    if signed_improvement > 0:
        return "positive_repair"
    return "live_no_improvement"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Exp 857: GGUFCacheResolver download + 25 HumanEval code repair v6."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE, requires_gpu=True)
    tmpl.setup()
    writer = AtomicResultWriter(DELIVERABLE)
    watchdog = ExperimentTimeoutWatchdog(TIMEOUT_MINUTES * 60)

    # Step 1: load session env so CARNOT_FORCE_LIVE propagates across process boundaries.
    EnvPropagationGuard.load_session_env()

    # Step 2: gate checks.
    if not check_exp855_gate():
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked",
                "blocked_by": "exp855_live_env_not_fixed",
                "blocked_reason": "Exp 855 did not set live_env_fixed=True.",
                "inference_mode": "blocked",
                "signed_improvement": None,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    if not check_exp856_gate():
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked",
                "blocked_by": "exp856_dual_gpu_not_deployed",
                "blocked_reason": "Exp 856 did not set dual_gpu_deployed=True.",
                "inference_mode": "blocked",
                "signed_improvement": None,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 3: determine inference mode.  EnvPropagationGuard.load_session_env()
    # should have set CARNOT_FORCE_LIVE=1 above.  LiveGPUGate.is_live() reads it.
    inference_mode = "live_gpu" if LiveGPUGate.check_env_var() else "simulated"
    if inference_mode != "live_gpu":
        artifact = tmpl.build_result(
            {
                "honest_verdict": "simulation_fallback",
                "inference_mode": inference_mode,
                "signed_improvement": None,
                "n_problems": N_PROBLEMS,
                "n_baseline_pass": 0,
                "n_repair_pass": 0,
                "note": (
                    "CARNOT_FORCE_LIVE not set — session env not propagated. "
                    "Exp 855 EnvPropagationGuard must be active for live GPU run."
                ),
            },
            status="success",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 4: set CARNOT_DUAL_GPU for dual-GPU wiring.
    os.environ["CARNOT_DUAL_GPU"] = "1"

    # Step 5: resolve model path (auto-downloads if absent — Exp 857 extension).
    resolver = GGUFCacheResolver(GGUFCacheConfig(cache_dir="models/", default_quantization=QUANTIZATION))
    try:
        model_path = resolver.resolve(MODEL_ID, QUANTIZATION)
    except GGUFModelNotFoundError as exc:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked",
                "blocked_by": "model_not_cached_download_failed",
                "blocked_reason": str(exc),
                "inference_mode": inference_mode,
                "signed_improvement": None,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 6: load model.
    try:
        from llama_cpp import Llama  # type: ignore[import]
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
    except Exception as exc:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked",
                "blocked_by": "model_load_failed",
                "blocked_reason": str(exc),
                "inference_mode": inference_mode,
                "model_path": str(model_path),
                "signed_improvement": None,
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step 7: run 25 problems in batches.
    extractor = CodeExtractor()
    problems = _INLINE_PROBLEMS[:N_PROBLEMS]
    n_baseline_pass = 0
    n_repair_pass = 0
    n_repairs_attempted = 0
    n_repairs_successful = 0
    problem_results: list[dict[str, Any]] = []

    for batch_idx in range(0, N_PROBLEMS, BATCH_SIZE):
        batch = problems[batch_idx : batch_idx + BATCH_SIZE]
        for prob in batch:
            if not watchdog.is_active():
                break

            baseline_pass = run_problem_baseline(prob)
            repair_pass, repair_attempted, logit_margin = run_problem_with_repair(
                prob, llm, extractor
            )

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
            {
                "n_baseline_pass": n_baseline_pass,
                "n_repair_pass": n_repair_pass,
                "problem_results": problem_results,
            },
            step=batch_idx + BATCH_SIZE,
        )

    n_live = len(problem_results)
    signed_improvement = compute_signed_improvement(n_repair_pass, n_baseline_pass, n_live)
    verdict = classify_verdict(signed_improvement, inference_mode)

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
            "problem_results": problem_results,
        },
        status="success",
        decision_class="repair",
    )
    writer.write(artifact)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
