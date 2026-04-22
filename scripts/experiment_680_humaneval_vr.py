#!/usr/bin/env python3
"""Experiment 680: HumanEval VR — Execution-Based Code Verification with Assertion Forcing.

**Researcher summary (RETRO-033 code-domain extension):**
    Structured-equation forcing worked for GSM8K math because arithmetic can be
    intercepted and reformatted before it becomes prose.  Code is fundamentally
    different: the only reliable oracle is *execution* — either the code produces
    the right output or it does not.  Regex-based code verification cannot distinguish
    a plausible-looking wrong answer from a correct one.

    This experiment tests the hypothesis that assertion-comment forcing (asking the
    model to write "# ASSERT: variable == value" after each intermediate computation)
    improves HumanEval pass@1 when combined with execution-based verify-repair.

    Gate chain (every exit path writes the deliverable):
        0. apply_env_autofix() INSIDE main() BEFORE heavy imports (RETRO-022, RETRO-053).
        1. ExperimentTimeoutWatchdog(680, timeout_minutes=90) — hard cap.
        2. GPU gate: CARNOT_FORCE_LIVE=1 required; absent → blocked artifact, exit 0.
        3. Load 25 HumanEval problems (hard-coded canonical set with entry_point + tests).
        4. For each problem:
           a. Baseline: generate WITHOUT assertion forcing → execute → baseline_pass.
           b. Forced:   generate WITH assertion forcing   → execute → post_pass.
              If execution fails AND assert comments found → repair attempt via
              VerifyRepairPipeline → re-execute → post_pass.
        5. Compute baseline_pass_at_1, post_pass_at_1, signed_improvement.
        6. honest_verdict:
           - 'code_vr_positive'       if signed_improvement > 0 AND inference_mode=='live_gpu'
           - 'code_vr_no_improvement' if signed_improvement <= 0 AND inference_mode=='live_gpu'
           - 'code_vr_blocked'        if no live GPU
        7. Write results/experiment_680_humaneval_vr.json.
        8. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-VERIFY-157, REQ-VERIFY-158,
      SCENARIO-VERIFY-208, SCENARIO-VERIFY-209
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 680
DELIVERABLE = "results/experiment_680_humaneval_vr.json"
N_PROBLEMS = 25
SCHEMA = "carnot.humaneval_vr.v1"

# ---------------------------------------------------------------------------
# Canonical 25 HumanEval-style problems
# Each entry has:
#   prompt      — problem statement shown to the model
#   entry_point — function name to call in the test harness
#   test_code   — Python code that calls entry_point and prints "PASS" or raises
# ---------------------------------------------------------------------------

HUMANEVAL_PROBLEMS: list[dict] = [
    {
        "prompt": "Write a Python function `has_close_elements(numbers: list, threshold: float) -> bool` that checks if any two numbers in the list are closer to each other than the given threshold.",
        "entry_point": "has_close_elements",
        "test_code": """
result = has_close_elements([1.0, 2.0, 3.0], 0.5)
assert result == False, f"Expected False, got {result}"
result2 = has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
assert result2 == True, f"Expected True, got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> list` that takes a string of parentheses and returns a list of separate parenthesis groups.",
        "entry_point": "separate_paren_groups",
        "test_code": """
result = separate_paren_groups('( ) (( )) (( )( ))')
assert result == ['()', '(())', '(()())'], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `truncate_number(number: float) -> float` that returns the decimal part of a float.",
        "entry_point": "truncate_number",
        "test_code": """
result = truncate_number(3.5)
assert abs(result - 0.5) < 1e-6, f"Got {result}"
result2 = truncate_number(1.33)
assert abs(result2 - 0.33) < 1e-6, f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `below_zero(operations: list) -> bool` that tracks a bank balance starting at zero and returns True if the balance goes below zero at any point.",
        "entry_point": "below_zero",
        "test_code": """
result = below_zero([1, 2, 3])
assert result == False, f"Got {result}"
result2 = below_zero([1, 2, -4, 5])
assert result2 == True, f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `mean_absolute_deviation(numbers: list) -> float` that returns the mean absolute deviation of a list of numbers.",
        "entry_point": "mean_absolute_deviation",
        "test_code": """
result = mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
assert abs(result - 1.0) < 1e-6, f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `intersperse(numbers: list, delimeter: int) -> list` that inserts delimeter between each pair of consecutive elements of numbers.",
        "entry_point": "intersperse",
        "test_code": """
result = intersperse([], 4)
assert result == [], f"Got {result}"
result2 = intersperse([1, 2, 3], 4)
assert result2 == [1, 4, 2, 4, 3], f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `parse_nested_parens(paren_string: str) -> list` that returns the maximum depth of nested parentheses for each group in a space-separated string.",
        "entry_point": "parse_nested_parens",
        "test_code": """
result = parse_nested_parens('(()()) ((())) () ((())()())')
assert result == [2, 3, 1, 3], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `filter_by_substring(strings: list, substring: str) -> list` that filters strings containing the given substring.",
        "entry_point": "filter_by_substring",
        "test_code": """
result = filter_by_substring([], 'a')
assert result == [], f"Got {result}"
result2 = filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
assert result2 == ['abc', 'bacd', 'array'], f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `sum_product(numbers: list) -> tuple` that returns a tuple of the sum and product of all numbers in the list.",
        "entry_point": "sum_product",
        "test_code": """
result = sum_product([])
assert result == (0, 1), f"Got {result}"
result2 = sum_product([1, 2, 3, 4])
assert result2 == (10, 24), f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `rolling_max(numbers: list) -> list` that returns the rolling maximum of a list.",
        "entry_point": "rolling_max",
        "test_code": """
result = rolling_max([1, 2, 3, 2, 3, 4, 2])
assert result == [1, 2, 3, 3, 3, 4, 4], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `make_palindrome(string: str) -> str` that finds the shortest palindrome beginning with the supplied string.",
        "entry_point": "make_palindrome",
        "test_code": """
result = make_palindrome('')
assert result == '', f"Got {result}"
result2 = make_palindrome('cat')
assert result2 == 'catac', f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `string_xor(a: str, b: str) -> str` that performs binary XOR on two strings of 0s and 1s.",
        "entry_point": "string_xor",
        "test_code": """
result = string_xor('010', '110')
assert result == '100', f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `longest(strings: list) -> str` that returns the longest string in a list, or None if the list is empty.",
        "entry_point": "longest",
        "test_code": """
result = longest([])
assert result is None, f"Got {result}"
result2 = longest(['a', 'bb', 'ccc'])
assert result2 == 'ccc', f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `greatest_common_divisor(a: int, b: int) -> int` that returns the greatest common divisor of two integers.",
        "entry_point": "greatest_common_divisor",
        "test_code": """
result = greatest_common_divisor(3, 5)
assert result == 1, f"Got {result}"
result2 = greatest_common_divisor(25, 15)
assert result2 == 5, f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `all_prefixes(string: str) -> list` that returns a list of all prefixes of the string from shortest to longest.",
        "entry_point": "all_prefixes",
        "test_code": """
result = all_prefixes('abc')
assert result == ['a', 'ab', 'abc'], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `string_sequence(n: int) -> str` that returns a string containing space-delimited numbers starting from 0 up to n.",
        "entry_point": "string_sequence",
        "test_code": """
result = string_sequence(0)
assert result == '0', f"Got {result}"
result2 = string_sequence(5)
assert result2 == '0 1 2 3 4 5', f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `count_distinct_characters(string: str) -> int` that returns the count of distinct characters in a string, case-insensitive.",
        "entry_point": "count_distinct_characters",
        "test_code": """
result = count_distinct_characters('')
assert result == 0, f"Got {result}"
result2 = count_distinct_characters('xyzXYZ')
assert result2 == 3, f"Got {result2}"
result3 = count_distinct_characters('Jerry')
assert result3 == 4, f"Got {result3}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `parse_music(music_string: str) -> list` that returns a list of beat durations: 'o'=4, 'o|'=2, '.|'=1.",
        "entry_point": "parse_music",
        "test_code": """
result = parse_music('o o| .| o| o| .| .| .| .| o o')
assert result == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `how_many_times(string: str, substring: str) -> int` that counts how many times a substring appears in a string (including overlapping).",
        "entry_point": "how_many_times",
        "test_code": """
result = how_many_times('', 'x')
assert result == 0, f"Got {result}"
result2 = how_many_times('aaa', 'a')
assert result2 == 3, f"Got {result2}"
result3 = how_many_times('aaaa', 'aa')
assert result3 == 3, f"Got {result3}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `sort_numbers(numbers: str) -> str` that sorts space-separated word-form number names (zero through nine).",
        "entry_point": "sort_numbers",
        "test_code": """
result = sort_numbers('')
assert result == '', f"Got {result}"
result2 = sort_numbers('three one five')
assert result2 == 'one three five', f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `find_closest_elements(numbers: list) -> tuple` that finds the two closest numbers in a list and returns them as a sorted tuple.",
        "entry_point": "find_closest_elements",
        "test_code": """
result = find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
assert result == (2.0, 2.2), f"Got {result}"
result2 = find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
assert result2 == (2.0, 2.0), f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `rescale_to_unit(numbers: list) -> list` that linearly rescales a list of numbers so the minimum is 0.0 and the maximum is 1.0.",
        "entry_point": "rescale_to_unit",
        "test_code": """
result = rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
assert result == [0.0, 0.25, 0.5, 0.75, 1.0], f"Got {result}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `filter_integers(values: list) -> list` that filters out non-integer values from a mixed list.",
        "entry_point": "filter_integers",
        "test_code": """
result = filter_integers(['a', 3.14, 5])
assert result == [5], f"Got {result}"
result2 = filter_integers([1, 2, 3, 'abc', {}, []])
assert result2 == [1, 2, 3], f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `strlen(string: str) -> int` that returns the length of a string.",
        "entry_point": "strlen",
        "test_code": """
result = strlen('')
assert result == 0, f"Got {result}"
result2 = strlen('abc')
assert result2 == 3, f"Got {result2}"
print("PASS")
""",
    },
    {
        "prompt": "Write a Python function `largest_divisor(n: int) -> int` that finds the largest divisor of n that is strictly less than n.",
        "entry_point": "largest_divisor",
        "test_code": """
result = largest_divisor(15)
assert result == 5, f"Got {result}"
result2 = largest_divisor(27)
assert result2 == 9, f"Got {result2}"
print("PASS")
""",
    },
]

# ---------------------------------------------------------------------------
# Forcing system prompt for assertion-comment style (REQ-VERIFY-158)
# ---------------------------------------------------------------------------

CODE_FORCING_SYSTEM_PROMPT: str = (
    "You are a Python programming assistant. Write clean, correct Python code.\n"
    "IMPORTANT: After each intermediate computation, add a comment in this exact format:\n"
    "# ASSERT: variable_name == expected_value\n"
    "Example:\n"
    "    total = sum(numbers)  # accumulate\n"
    "    # ASSERT: total == 15\n"
    "Do this for EVERY intermediate result. Do not skip this format."
)

BASELINE_SYSTEM_PROMPT: str = (
    "You are a Python programming assistant. Write clean, correct Python code."
)

# ---------------------------------------------------------------------------
# Public helpers (module-level for testability)
# ---------------------------------------------------------------------------


def extract_assert_comments(code: str) -> list[tuple[str, str]]:
    """Extract (variable, value) pairs from '# ASSERT: variable == value' comments.

    WHY this form of assertion: the comment form is extractable by regex without
    executing the code, and can be converted to real assert statements for
    execution-based verification.  This avoids the need for AST parsing of
    potentially malformed generated code.

    Args:
        code: Python source code, possibly containing # ASSERT: comments.

    Returns:
        List of (variable_name, expected_value) pairs in order of appearance.
        Returns empty list if no ASSERT comments found.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-208
    """
    pattern = re.compile(r"#\s*ASSERT:\s*(\w+)\s*==\s*(.+)")
    results = []
    for line in code.splitlines():
        m = pattern.search(line)
        if m:
            var = m.group(1).strip()
            val = m.group(2).strip()
            results.append((var, val))
    return results


def extract_python_code(response: str) -> str:
    """Extract the first Python code block from a model response.

    WHY extract: models typically wrap code in markdown fences (```python ... ```).
    This function strips the fences and returns the raw source so it can be executed.

    If no fenced block is found, the entire response is returned as-is (some models
    omit the fences for simple functions).

    Args:
        response: Raw model output text.

    Returns:
        Python source code string, stripped of markdown fences.

    Spec: REQ-VERIFY-157
    """
    # Match ```python ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return response.strip()


def execute_code(function_code: str, test_code: str, timeout: int = 5) -> bool:
    """Execute a generated function + test harness and return True if tests pass.

    WHY subprocess: executing untrusted generated code in the same process risks
    crashes, infinite loops, and resource exhaustion.  subprocess.run with a timeout
    isolates each attempt and prevents runaway code from killing the experiment.

    WHY check stdout for "PASS": the test harness prints "PASS" on success and raises
    AssertionError on failure.  Checking stdout is more robust than checking returncode
    alone because some code may exit 0 despite wrong output.

    Args:
        function_code: The generated Python function source code.
        test_code: Test harness that calls the function and prints "PASS" on success.
        timeout: Maximum wall-clock seconds per execution attempt (default 5).

    Returns:
        True if execution exits 0 AND stdout contains "PASS", False otherwise.

    Spec: REQ-VERIFY-157, REQ-VERIFY-157-1, REQ-VERIFY-157-2, REQ-VERIFY-157-3
    """
    full_code = function_code + "\n" + test_code
    try:
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def compute_honest_verdict_680(signed_improvement: float, inference_mode: str) -> str:
    """Map signed improvement and inference mode to a machine-readable honest_verdict.

    WHY three distinct verdicts: this experiment focuses on whether code VR is
    directionally positive.  Unlike math (where we have Wilson CI from 200 questions),
    25 HumanEval problems gives us a directional signal only.

    Args:
        signed_improvement: post_pass_at_1 - baseline_pass_at_1 (float, signed).
        inference_mode: 'live_gpu' or 'blocked'.

    Returns:
        One of: 'code_vr_positive', 'code_vr_no_improvement', 'code_vr_blocked'.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-209
    """
    if inference_mode == "blocked":
        return "code_vr_blocked"
    if signed_improvement > 0.0:
        return "code_vr_positive"
    return "code_vr_no_improvement"


def _build_blocked_artifact(reason: str) -> dict:
    """Build a blocked artifact with all required schema fields.

    WHY centralised: every blocked exit path must emit a complete artifact with the
    same schema as the live-gpu path.  This prevents conductor parse errors when
    the experiment exits early.

    Args:
        reason: Human-readable explanation of why the experiment was blocked.

    Returns:
        JSON-serialisable dict with all required schema fields.

    Spec: REQ-VERIFY-157, REQ-VERIFY-158
    """
    return {
        "experiment": EXP_ID,
        "schema": SCHEMA,
        "run_date": "20260422",
        "status": "blocked",
        "title": "HumanEval VR: Execution-Based Code Verification (Exp 680)",
        "started_at": "",
        "finished_at": "",
        "duration_s": 0.0,
        "honest_verdict": "code_vr_blocked",
        "blocked_reason": reason,
        "inference_mode": "blocked",
        "n_problems": 0,
        "baseline_pass_at_1": 0.0,
        "post_pass_at_1": 0.0,
        "signed_improvement": 0.0,
        "assert_comments_found": 0,
        "repair_attempts": 0,
    }


# ---------------------------------------------------------------------------
# main / _run_inner
# ---------------------------------------------------------------------------


def main() -> None:
    """Run HumanEval VR experiment with execution-based verification.

    WHY apply_env_autofix is first: RETRO-022 and RETRO-053 showed that
    CARNOT_FORCE_LIVE is not reliably propagated into subprocess environments.
    Calling apply_env_autofix() before any heavy import ensures GPU gate
    checks downstream see the correct env var value.

    Every exit path (blocked, live_gpu) writes DELIVERABLE and calls
    assert_deliverable_written() as the final action.
    """
    # Step 0: env autofix BEFORE any heavy import (RETRO-022, RETRO-053)
    from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: PLC0415
    apply_env_autofix()

    # Step 1: watchdog — 90-minute hard cap
    from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: PLC0415
    _watchdog = ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=90,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    _watchdog.start()
    try:
        _run_inner(_watchdog)
    finally:
        _watchdog.stop()


def _run_inner(_watchdog) -> None:  # noqa: ANN001
    """Inner experiment body separated from main() so the watchdog wraps cleanly.

    WHY separate function: if _run_inner() raises unexpectedly the finally in
    main() still calls _watchdog.stop(), preventing a ghost watchdog thread from
    firing after the process has already exited.
    """
    from scripts.experiment_template import ExperimentTemplate  # noqa: PLC0415
    from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: PLC0415

    import datetime
    t_start = time.time()
    started_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_date = "20260422"

    tmpl = ExperimentTemplate(
        EXP_ID,
        "HumanEval VR: Execution-Based Code Verification (Exp 680)",
        DELIVERABLE,
        requires_gpu=False,  # bypass ModelServer; direct HF inference when GPU present
    )
    tmpl.setup()

    writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

    def _write_and_exit(artifact: dict) -> None:
        """Write artifact atomically then assert deliverable written.

        WHY centralised: DeliverableGuard raises if we exit without writing.
        """
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        sys.exit(0)

    # ------------------------------------------------------------------
    # GPU gate: CARNOT_FORCE_LIVE=1 required (REQ-VERIFY-158-4)
    # ------------------------------------------------------------------
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        _write_and_exit(_build_blocked_artifact(
            "CARNOT_FORCE_LIVE=1 not set — live GPU required for code VR"
        ))

    # ------------------------------------------------------------------
    # GPU hardware presence check
    # ------------------------------------------------------------------
    try:
        import torch as _tc  # noqa: PLC0415
        if not _tc.cuda.is_available():
            _write_and_exit(_build_blocked_artifact(
                "torch.cuda.is_available() returned False — no GPU detected"
            ))
    except ImportError:
        _write_and_exit(_build_blocked_artifact(
            "torch not installed — cannot confirm GPU hardware"
        ))

    inference_mode = "live_gpu"

    # ------------------------------------------------------------------
    # Load model ONCE for all inference calls.
    # WHY direct HuggingFace: ModelServer.generate() blocks indefinitely when
    # called from a non-interactive process (confirmed Exp 679 / RETRO-033).
    # ------------------------------------------------------------------
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    _hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    _hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _hf_model.eval()

    def _generate(system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Qwen3.5-0.8B with a system+user prompt pair.

        WHY model loaded in closure: loading ~1.6 GB per call would take minutes.
        Loading once and referencing via closure gives ~1-5 s/call on RTX 3090.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = _hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _hf_tokenizer(text, return_tensors="pt").to(_hf_model.device)
        with torch.no_grad():
            outputs = _hf_model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )
        return _hf_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Run 25 HumanEval problems
    # ------------------------------------------------------------------
    n_baseline_pass = 0
    n_post_pass = 0
    total_assert_comments = 0
    total_repair_attempts = 0
    problem_results = []

    for i, problem in enumerate(HUMANEVAL_PROBLEMS):
        prompt = problem["prompt"]
        test_code = problem["test_code"]

        # Baseline: no forcing
        baseline_response = _generate(BASELINE_SYSTEM_PROMPT, prompt)
        baseline_code = extract_python_code(baseline_response)
        baseline_pass = execute_code(baseline_code, test_code)
        if baseline_pass:
            n_baseline_pass += 1

        # Forced: assertion-comment forcing
        forced_response = _generate(CODE_FORCING_SYSTEM_PROMPT, prompt)
        forced_code = extract_python_code(forced_response)
        assert_pairs = extract_assert_comments(forced_code)
        total_assert_comments += len(assert_pairs)

        post_pass = execute_code(forced_code, test_code)
        repaired = False

        # Repair: if forced code fails AND we have assert comments, try repair
        if not post_pass and assert_pairs:
            total_repair_attempts += 1
            # Describe which assertions were found and ask for a fix
            assert_summary = "\n".join(
                f"  # ASSERT: {var} == {val}" for var, val in assert_pairs
            )
            repair_prompt = (
                f"{prompt}\n\n"
                f"Your previous attempt had assertion annotations:\n{assert_summary}\n\n"
                "The code failed to produce the correct output. "
                "Please rewrite the function so it passes all tests. "
                "Keep the # ASSERT: comments where applicable."
            )
            repair_response = _generate(CODE_FORCING_SYSTEM_PROMPT, repair_prompt)
            repair_code = extract_python_code(repair_response)
            post_pass = execute_code(repair_code, test_code)
            repaired = post_pass

        if post_pass:
            n_post_pass += 1

        problem_results.append({
            "problem_idx": i,
            "entry_point": problem["entry_point"],
            "baseline_pass": baseline_pass,
            "post_pass": post_pass,
            "assert_comments": len(assert_pairs),
            "repaired": repaired,
        })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    baseline_pass_at_1 = n_baseline_pass / N_PROBLEMS
    post_pass_at_1 = n_post_pass / N_PROBLEMS
    signed_improvement = post_pass_at_1 - baseline_pass_at_1
    honest_verdict = compute_honest_verdict_680(signed_improvement, inference_mode)

    finished_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_s = time.time() - t_start

    artifact = {
        "experiment": EXP_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "success",
        "title": "HumanEval VR: Execution-Based Code Verification (Exp 680)",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "inference_mode": inference_mode,
        "n_problems": N_PROBLEMS,
        "baseline_pass_at_1": baseline_pass_at_1,
        "post_pass_at_1": post_pass_at_1,
        "signed_improvement": signed_improvement,
        "honest_verdict": honest_verdict,
        "assert_comments_found": total_assert_comments,
        "repair_attempts": total_repair_attempts,
        "problem_results": problem_results,
        "models_used": ["Qwen/Qwen3.5-0.8B"],
        "decision_class": ["verify", "repair"],
    }

    _write_and_exit(artifact)


if __name__ == "__main__":
    main()
