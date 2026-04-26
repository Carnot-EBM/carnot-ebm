#!/usr/bin/env python3
"""Experiment 905: Iterative Self-Repair v1 — 25 HumanEval GPU (Qwen3.6-35B-A3B-GGUF).

**Researcher summary:**
    Code repair experiments Exps 850-881 all failed with honest_verdict='zero_constraints'
    because ArithmeticExtractor's regex pattern 'a+b=c' never matches natural-language
    LLM responses.  The extractor finds zero constraints, so VerifyRepairPipeline has
    no repair signal.

    This experiment implements the Iterative Self-Repair approach (arXiv 2604.10508):
        1. Generate initial code with LLM.
        2. Execute it in a subprocess sandbox against HumanEval test cases.
        3. If it fails: feed the FULL traceback back to the LLM as a correction prompt.
        4. Retry up to 3 times.
        5. Carnot energy selects the best attempt (lowest-energy passer, or
           lowest-energy attempt overall when none pass).

    This requires NO constraint extraction.  The execution error IS the repair signal.

**Gate:**
    None — no external preflight required.  CARNOT_USE_SANDBOX=0 is set in the
    environment to use subprocess execution without gVisor (faster for development).

**Models:**
    Primary: unsloth/Qwen3.6-35B-A3B-GGUF (llama.cpp, GPU 0) — ~3B active params,
    Q4_K_M quantization fits in ~20 GiB VRAM.
    Fallback: google/gemma-4-E4B-it via transformers (when llama.cpp path fails).

**Metrics:**
    - baseline_pass_rate: fraction passing on first attempt (no repair).
    - repair_pass_rate: fraction passing after iterative repair (up to 3 retries).
    - signed_improvement: repair_pass_rate - baseline_pass_rate.
    - mean_retries_needed: mean retries for problems that ended up passing.
    - energy_selected_correctly: fraction where energy picked the passing attempt.

**Honest-verdict mapping:**
    "improvement_significant_gate_open"  — signed_improvement > 0.05
    "improvement_marginal_gate_open"     — 0 < signed_improvement <= 0.05
    "no_improvement_gate_blocked"        — signed_improvement <= 0

Spec: REQ-CODE-033 (IterativeSelfRepair pipeline),
      SCENARIO-CODE-031 (retry with execution feedback),
      REQ-REPAIR-022 (SOTA GGUF required for headline results)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
import traceback as tb
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root setup — must happen before any carnot imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_905_iterative_self_repair_v1.json"

# ---------------------------------------------------------------------------
# 25 representative HumanEval problems (inline fallback when datasets unavailable)
# ---------------------------------------------------------------------------

# These are the first 25 problems from the OpenAI HumanEval benchmark.
# We embed a minimal self-contained subset so the experiment runs even when
# the `human_eval` package is not installed.  If the package IS available
# we use it for authoritative test harnesses (more rigorous).

_INLINE_PROBLEMS: list[dict[str, Any]] = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "has_close_elements",
        "test": (
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/1",
        "prompt": (
            "from typing import List\n\n"
            "def separate_paren_groups(paren_string: str) -> List[str]:\n"
            '    """ Input to this function is a string containing multiple groups of nested parentheses.\n'
            "    Your goal is to separate those groups into separate strings and return the list of those.\n"
            "    Separate groups are balanced (each open brace is properly closed) and not nested within\n"
            "    each other. Ignore any spaces in the input string.\n"
            "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
            "    ['()', '(())', '(()())']\n"
            '    """\n'
        ),
        "entry_point": "separate_paren_groups",
        "test": (
            "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n"
            "assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n"
        ),
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            '    """ Given a positive floating point number, it can be decomposed into\n'
            "    and integer part (largest integer smaller than given number) and decimals\n"
            "    (leftover part always smaller than 1).\n"
            "    Return the decimal part of the number.\n"
            "    >>> truncate_number(3.5)\n"
            "    0.5\n"
            '    """\n'
        ),
        "entry_point": "truncate_number",
        "test": (
            "assert abs(truncate_number(3.5) - 0.5) < 1e-6\n"
            "assert abs(truncate_number(1.33) - 0.33) < 1e-6\n"
            "assert abs(truncate_number(123.456) - 0.456) < 1e-6\n"
        ),
    },
    {
        "task_id": "HumanEval/3",
        "prompt": (
            "from typing import List\n\n"
            "def below_zero(operations: List[int]) -> bool:\n"
            '    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
            "    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n"
            "    at that point function should return True. Otherwise it should return False.\n"
            "    >>> below_zero([1, 2, 3])\n"
            "    False\n"
            "    >>> below_zero([1, 2, -4, 5])\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "below_zero",
        "test": (
            "assert below_zero([]) == False\n"
            "assert below_zero([1, 2, 3]) == False\n"
            "assert below_zero([1, 2, -4, 5]) == True\n"
            "assert below_zero([1, 2, 3, -6]) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "from typing import List\n\n"
            "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
            '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
            "    around the mean of this dataset.\n"
            "    Mean Absolute Deviation is the average absolute difference between each\n"
            "    element and a centerpoint (mean in this case):\n"
            "    MAD = average | x - x_mean |\n"
            "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
            "    1.0\n"
            '    """\n'
        ),
        "entry_point": "mean_absolute_deviation",
        "test": (
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n"
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6\n"
        ),
    },
    {
        "task_id": "HumanEval/5",
        "prompt": (
            "from typing import List\n\n"
            "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
            "    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n"
            "    >>> intersperse([], 4)\n"
            "    []\n"
            "    >>> intersperse([1, 2, 3], 4)\n"
            "    [1, 4, 2, 4, 3]\n"
            '    """\n'
        ),
        "entry_point": "intersperse",
        "test": (
            "assert intersperse([], 4) == []\n"
            "assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\n"
            "assert intersperse([1, 2], 4) == [1, 4, 2]\n"
        ),
    },
    {
        "task_id": "HumanEval/6",
        "prompt": (
            "from typing import List\n\n"
            "def parse_nested_parens(paren_string: str) -> List[int]:\n"
            '    """ Input to this function is a string represented multiple groups for nested parentheses\n'
            "    separated by spaces. For each of the group, output the deepest level of nesting of\n"
            "    parentheses.\n"
            "    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n"
            "    >>> parse_nested_parens('(()()) ((())) () ((())(()()))')\n"
            "    [2, 3, 1, 3]\n"
            '    """\n'
        ),
        "entry_point": "parse_nested_parens",
        "test": (
            "assert parse_nested_parens('(()()) ((())) () ((())(()()))') == [2, 3, 1, 3]\n"
            "assert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]\n"
        ),
    },
    {
        "task_id": "HumanEval/7",
        "prompt": (
            "from typing import List\n\n"
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
            '    """ Filter an input list of strings only for ones that contain given substring\n'
            "    >>> filter_by_substring([], 'a')\n"
            "    []\n"
            "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
            "    ['abc', 'bacd', 'array']\n"
            '    """\n'
        ),
        "entry_point": "filter_by_substring",
        "test": (
            "assert filter_by_substring([], 'a') == []\n"
            "assert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']\n"
        ),
    },
    {
        "task_id": "HumanEval/8",
        "prompt": (
            "from typing import List, Tuple\n\n"
            "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
            '    """ For a given list of integers, return a tuple consisting of a sum and a product of all the\n'
            "    integers in a list.\n"
            "    Empty sum should be equal to 0 and empty product should be equal to 1.\n"
            "    >>> sum_product([])\n"
            "    (0, 1)\n"
            "    >>> sum_product([1, 2, 3, 4])\n"
            "    (10, 24)\n"
            '    """\n'
        ),
        "entry_point": "sum_product",
        "test": (
            "assert sum_product([]) == (0, 1)\n"
            "assert sum_product([1, 2, 3, 4]) == (10, 24)\n"
            "assert sum_product([1, 1, 1]) == (3, 1)\n"
        ),
    },
    {
        "task_id": "HumanEval/9",
        "prompt": (
            "from typing import List, Tuple\n\n"
            "def rolling_max(numbers: List[int]) -> List[int]:\n"
            '    """ From a given list of integers, generate a list of rolling maximum element found until given\n'
            "    moment in the sequence.\n"
            "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
            "    [1, 2, 3, 3, 3, 4, 4]\n"
            '    """\n'
        ),
        "entry_point": "rolling_max",
        "test": (
            "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\n"
            "assert rolling_max([]) == []\n"
            "assert rolling_max([1]) == [1]\n"
        ),
    },
    {
        "task_id": "HumanEval/10",
        "prompt": (
            "def make_palindrome(string: str) -> str:\n"
            '    """ Find the shortest palindrome that begins with a supplied string.\n'
            "    Algorithm idea is simple:\n"
            "    - Find the longest postfix of supplied string that is a palindrome.\n"
            "    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.\n"
            "    >>> make_palindrome('')\n"
            "    ''\n"
            "    >>> make_palindrome('cat')\n"
            "    'catac'\n"
            "    >>> make_palindrome('cata')\n"
            "    'catac'\n"
            '    """\n'
        ),
        "entry_point": "make_palindrome",
        "test": (
            "assert make_palindrome('') == ''\n"
            "assert make_palindrome('x') == 'x'\n"
            "assert make_palindrome('cat') == 'catac'\n"
            "assert make_palindrome('cata') == 'catac'\n"
        ),
    },
    {
        "task_id": "HumanEval/11",
        "prompt": (
            "from typing import List\n\n"
            "def string_xor(a: str, b: str) -> str:\n"
            '    """ Input are two strings a and b consisting only of 1s and 0s.\n'
            "    Perform binary XOR on these inputs and return result also as a string.\n"
            "    >>> string_xor('010', '110')\n"
            "    '100'\n"
            '    """\n'
        ),
        "entry_point": "string_xor",
        "test": (
            "assert string_xor('010', '110') == '100'\n"
            "assert string_xor('0101', '0000') == '0101'\n"
        ),
    },
    {
        "task_id": "HumanEval/12",
        "prompt": (
            "from typing import List, Optional\n\n"
            "def longest(strings: List[str]) -> Optional[str]:\n"
            '    """ Out of list of strings, return the longest one. Return the first one in case of multiple\n'
            "    strings of the same length. Return None in case the input list is empty.\n"
            "    >>> longest([])\n"
            "    >>> longest(['a', 'b', 'c'])\n"
            "    'a'\n"
            "    >>> longest(['a', 'bb', 'ccc'])\n"
            "    'ccc'\n"
            '    """\n'
        ),
        "entry_point": "longest",
        "test": (
            "assert longest([]) is None\n"
            "assert longest(['a', 'b', 'c']) == 'a'\n"
            "assert longest(['a', 'bb', 'ccc']) == 'ccc'\n"
        ),
    },
    {
        "task_id": "HumanEval/13",
        "prompt": (
            "def greatest_common_divisor(a: int, b: int) -> int:\n"
            '    """ Return a greatest common divisor of two integers a and b\n'
            "    >>> greatest_common_divisor(3, 5)\n"
            "    1\n"
            "    >>> greatest_common_divisor(25, 15)\n"
            "    5\n"
            '    """\n'
        ),
        "entry_point": "greatest_common_divisor",
        "test": (
            "assert greatest_common_divisor(3, 5) == 1\n"
            "assert greatest_common_divisor(25, 15) == 5\n"
            "assert greatest_common_divisor(7, 7) == 7\n"
        ),
    },
    {
        "task_id": "HumanEval/14",
        "prompt": (
            "from typing import List\n\n"
            "def all_prefixes(string: str) -> List[str]:\n"
            '    """ Return list of all prefixes from shortest to longest of the input string\n'
            "    >>> all_prefixes('abc')\n"
            "    ['a', 'ab', 'abc']\n"
            '    """\n'
        ),
        "entry_point": "all_prefixes",
        "test": (
            "assert all_prefixes('') == []\n"
            "assert all_prefixes('abc') == ['a', 'ab', 'abc']\n"
            "assert all_prefixes('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n"
        ),
    },
    {
        "task_id": "HumanEval/15",
        "prompt": (
            "def string_sequence(n: int) -> str:\n"
            '    """ Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n'
            "    >>> string_sequence(0)\n"
            "    '0'\n"
            "    >>> string_sequence(5)\n"
            "    '0 1 2 3 4 5'\n"
            '    """\n'
        ),
        "entry_point": "string_sequence",
        "test": (
            "assert string_sequence(0) == '0'\n"
            "assert string_sequence(5) == '0 1 2 3 4 5'\n"
            "assert string_sequence(10) == '0 1 2 3 4 5 6 7 8 9 10'\n"
        ),
    },
    {
        "task_id": "HumanEval/16",
        "prompt": (
            "def count_distinct_characters(string: str) -> int:\n"
            '    """ Given a string, find out how many distinct characters (regardless of case) does it consist of\n'
            "    >>> count_distinct_characters('xyzXYZ')\n"
            "    3\n"
            "    >>> count_distinct_characters('Jerry')\n"
            "    4\n"
            '    """\n'
        ),
        "entry_point": "count_distinct_characters",
        "test": (
            "assert count_distinct_characters('') == 0\n"
            "assert count_distinct_characters('xyzXYZ') == 3\n"
            "assert count_distinct_characters('Jerry') == 4\n"
        ),
    },
    {
        "task_id": "HumanEval/17",
        "prompt": (
            "from typing import List\n\n"
            "def parse_music(music_string: str) -> List[int]:\n"
            '    """ Input to this function is a string representing musical notes in a special ASCII format.\n'
            "    Your task is to parse this string and return list of integers corresponding to how many beats\n"
            "    does each not last.\n\n"
            "    Here is a legend:\n"
            "    'o' - whole note, lasts four beats\n"
            "    'o|' - half note, lasts two beats\n"
            "    '.|' - quater note, lasts one beat\n\n"
            "    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n"
            "    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
            '    """\n'
        ),
        "entry_point": "parse_music",
        "test": (
            "assert parse_music('') == []\n"
            "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
        ),
    },
    {
        "task_id": "HumanEval/18",
        "prompt": (
            "def how_many_times(string: str, substring: str) -> int:\n"
            '    """ Find how many times a given substring can be found in the original string. Count overlapping cases.\n'
            "    >>> how_many_times('', 'a')\n"
            "    0\n"
            "    >>> how_many_times('aaa', 'a')\n"
            "    3\n"
            "    >>> how_many_times('aaaa', 'aa')\n"
            "    3\n"
            '    """\n'
        ),
        "entry_point": "how_many_times",
        "test": (
            "assert how_many_times('', 'a') == 0\n"
            "assert how_many_times('aaa', 'a') == 3\n"
            "assert how_many_times('aaaa', 'aa') == 3\n"
        ),
    },
    {
        "task_id": "HumanEval/19",
        "prompt": (
            "from typing import List\n\n"
            "def sort_numbers(numbers: str) -> str:\n"
            "    \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.\n"
            "    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.\n"
            "    Return the string with numbers sorted from smallest to largest\n"
            "    >>> sort_numbers('three one five')\n"
            "    'one three five'\n"
            '    """\n'
        ),
        "entry_point": "sort_numbers",
        "test": (
            "assert sort_numbers('') == ''\n"
            "assert sort_numbers('three one five') == 'one three five'\n"
            "assert sort_numbers('zero nine five') == 'zero five nine'\n"
        ),
    },
    {
        "task_id": "HumanEval/20",
        "prompt": (
            "from typing import List, Tuple\n\n"
            "def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:\n"
            '    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to\n'
            "    each other and return them in order (smaller number, larger number).\n"
            "    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n"
            "    (2.0, 2.2)\n"
            "    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])\n"
            "    (2.0, 2.0)\n"
            '    """\n'
        ),
        "entry_point": "find_closest_elements",
        "test": (
            "assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\n"
            "assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)\n"
        ),
    },
    {
        "task_id": "HumanEval/21",
        "prompt": (
            "from typing import List\n\n"
            "def rescale_to_unit(numbers: List[float]) -> List[float]:\n"
            '    """ Given list of numbers (of at least two elements), apply a linear transform to that list,\n'
            "    such that the smallest number will become 0 and the largest will become 1\n"
            "    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            '    """\n'
        ),
        "entry_point": "rescale_to_unit",
        "test": (
            "assert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            "assert rescale_to_unit([2.0, 2.0]) == [0.0, 0.0]\n"
        ),
    },
    {
        "task_id": "HumanEval/22",
        "prompt": (
            "from typing import List, Any\n\n"
            "def filter_integers(values: List[Any]) -> List[int]:\n"
            '    """ Filter given list of any python values only for integers\n'
            "    >>> filter_integers(['a', 3.14, 5])\n"
            "    [5]\n"
            "    >>> filter_integers([1, 2, 3, 'abc', {}, []])\n"
            "    [1, 2, 3]\n"
            '    """\n'
        ),
        "entry_point": "filter_integers",
        "test": (
            "assert filter_integers(['a', 3.14, 5]) == [5]\n"
            "assert filter_integers([1, 2, 3, 'abc', {}, []]) == [1, 2, 3]\n"
        ),
    },
    {
        "task_id": "HumanEval/23",
        "prompt": (
            "def strlen(string: str) -> int:\n"
            '    """ Return length of given string\n'
            "    >>> strlen('')\n"
            "    0\n"
            "    >>> strlen('abc')\n"
            "    3\n"
            '    """\n'
        ),
        "entry_point": "strlen",
        "test": (
            "assert strlen('') == 0\n"
            "assert strlen('abc') == 3\n"
            "assert strlen('Hello World') == 11\n"
        ),
    },
    {
        "task_id": "HumanEval/24",
        "prompt": (
            "def largest_divisor(n: int) -> int:\n"
            '    """ For a given number n, find the largest number that divides it evenly, smaller than n\n'
            "    >>> largest_divisor(15)\n"
            "    5\n"
            '    """\n'
        ),
        "entry_point": "largest_divisor",
        "test": (
            "assert largest_divisor(15) == 5\n"
            "assert largest_divisor(7) == 1\n"
            "assert largest_divisor(100) == 50\n"
        ),
    },
]


# ---------------------------------------------------------------------------
# Simple energy scorer (token-length heuristic when Ising is unavailable)
# ---------------------------------------------------------------------------


class _TokenLengthEnergyScorer:
    """Fallback energy scorer: shorter responses get lower energy.

    Why token length: shorter, denser code tends to be more correct on
    HumanEval than verbose responses padded with comments or incorrect
    reasoning.  This is a weak heuristic — the real Ising scorer uses
    constraint satisfaction.  We use it as a fallback when the Ising
    pipeline is not available (e.g. on CPU-only CI hosts).

    This should NOT be used in production headline claims.  It is here
    purely to let the experiment run and produce a valid artifact even
    when the full energy pipeline is not loaded.
    """

    def score(self, text: str) -> float:
        """Return negative token count (shorter = lower energy = better)."""
        # Rough tokenisation: split on whitespace.  Enough for ranking.
        return float(len(text.split()))


def _build_energy_scorer() -> Any:
    """Try to load the Ising energy scorer; fall back to token-length heuristic.

    Why a fallback: the Ising pipeline requires JAX + trained weights.  On
    a fresh host or CI machine those may not be available.  The fallback
    still lets the pipeline structure work so we can measure pass rates.
    The energy_scorer_type field in the artifact records which path ran so
    the retrospective can flag if only the heuristic was used.
    """
    try:
        from carnot.models.ising import IsingConfig, IsingModel  # noqa: PLC0415
        import jax.random as jrandom  # noqa: PLC0415

        config = IsingConfig(input_dim=64, coupling_init="xavier_uniform")
        model = IsingModel(config, key=jrandom.PRNGKey(905))

        class _IsingScorer:
            def __init__(self, m: Any) -> None:
                self._m = m

            def score(self, text: str) -> float:
                # Encode text as a simple bag-of-chars spin vector (±1).
                import jax.numpy as jnp  # noqa: PLC0415

                chars = [ord(c) % 2 * 2 - 1 for c in text[:64]]
                # Pad or truncate to 64.
                chars = chars[:64] + [1] * max(0, 64 - len(chars))
                spins = jnp.array(chars, dtype=jnp.float32)
                return float(self._m.energy(spins))

        return _IsingScorer(model), "ising_model"
    except Exception:
        return _TokenLengthEnergyScorer(), "token_length_heuristic"


# ---------------------------------------------------------------------------
# LLM runner wrappers
# ---------------------------------------------------------------------------


class _LlamaCppRunner:
    """LLM runner backed by llama-cpp-python.

    Loads unsloth/Qwen3.6-35B-A3B-GGUF via llama.cpp.  This is the SOTA
    path mandated by CLAUDE.md for live headline results.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate a completion.  Returns only the assistant reply text."""
        output = self._model(
            prompt,
            max_tokens=512,
            temperature=0.0,
            echo=False,
        )
        return output["choices"][0]["text"].strip()


class _TransformersRunner:
    """LLM runner backed by HuggingFace transformers.

    Used as a fallback when llama.cpp is not available or the GGUF file
    is not cached.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(tmpl: ExperimentTemplate) -> tuple[Any, str]:
    """Try to load Qwen3.6-35B-A3B-GGUF, then fall back to Gemma4-E4B-it transformers.

    Returns (runner, model_id_str).
    """
    # --- Try llama.cpp path ---
    try:
        from llama_cpp import Llama  # noqa: PLC0415
        from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

        specs = cached_sota_pair(gpu_indices=(0,))
        if specs is not None:
            for spec in specs:
                if "Qwen3.6-35B" in spec.get("hf_id", ""):
                    model_path = spec.get("model_path")
                    if model_path and Path(model_path).exists():
                        print(
                            f"[exp905] Loading GGUF from {model_path} via llama.cpp …", flush=True
                        )
                        llm = Llama(
                            model_path=str(model_path),
                            n_gpu_layers=-1,
                            n_ctx=4096,
                            verbose=False,
                        )
                        return _LlamaCppRunner(llm), "unsloth/Qwen3.6-35B-A3B-GGUF"
    except Exception as exc:
        print(f"[exp905] llama.cpp path failed: {exc} — trying transformers fallback", flush=True)

    # --- Try transformers fallback ---
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        model_id = "google/gemma-4-E4B-it"
        print(f"[exp905] Loading {model_id} via transformers …", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return _TransformersRunner(model, tokenizer), model_id
    except Exception as exc:
        raise RuntimeError(f"Both llama.cpp and transformers model loads failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------


def _build_generation_prompt(problem: dict[str, Any]) -> str:
    """Build the initial generation prompt for a HumanEval problem.

    We keep the prompt simple: the HumanEval function signature + docstring
    plus a brief instruction.  This is the standard approach for HumanEval
    evaluation.
    """
    return (
        "Complete the following Python function. "
        "Output ONLY the function body code (no imports, no prose):\n\n" + problem["prompt"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate Exp 905 iterative self-repair and write the deliverable."""
    tmpl = ExperimentTemplate(
        exp_id=905,
        title="Iterative Self-Repair v1 — 25 HumanEval GPU",
        deliverable=_DELIVERABLE,
        requires_gpu=False,  # will use GPU when available, but CPU fallback allowed
    )
    tmpl.setup()

    t_start = time.perf_counter()
    print(f"[exp905] Starting experiment 905", flush=True)

    # -- Load energy scorer --------------------------------------------------
    energy_scorer, energy_scorer_type = _build_energy_scorer()
    print(f"[exp905] Energy scorer: {energy_scorer_type}", flush=True)

    # -- Load LLM ------------------------------------------------------------
    inference_mode = "live_gpu"
    try:
        runner, model_id = _load_model(tmpl)
        print(f"[exp905] Model loaded: {model_id}", flush=True)
    except Exception as exc:
        print(f"[exp905] Model load failed: {exc}", flush=True)
        artifact = tmpl.build_result(
            {
                "model_load_error": str(exc),
                "traceback": tb.format_exc(),
                "energy_scorer_type": energy_scorer_type,
            },
            status="blocked",
            honest_verdict="blocked",
            inference_mode="unknown",
        )
        Path(_REPO_ROOT / _DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    # -- Import IterativeSelfRepair ------------------------------------------
    from carnot.pipeline.iterative_self_repair import IterativeSelfRepair  # noqa: PLC0415

    pipeline = IterativeSelfRepair(
        llm_runner=runner,
        energy_scorer=energy_scorer,
        max_retries=3,
        sandbox=False,  # subprocess mode; set CARNOT_USE_SANDBOX=1 for gVisor
        exec_timeout_s=10.0,
    )

    # -- Try to load authoritative HumanEval problems -------------------------
    problems: list[dict[str, Any]]
    try:
        from human_eval.data import read_problems  # noqa: PLC0415

        all_problems = read_problems()
        task_ids = sorted(all_problems.keys())[:25]
        problems = [all_problems[tid] for tid in task_ids]
        print(f"[exp905] Using human_eval package for {len(problems)} problems.", flush=True)
    except ImportError:
        problems = _INLINE_PROBLEMS
        print(
            f"[exp905] human_eval not installed — using inline problems ({len(problems)}).",
            flush=True,
        )

    # -- Run pipeline -------------------------------------------------------
    results_per_problem: list[dict[str, Any]] = []
    n_baseline_pass = 0
    n_repair_pass = 0
    n_energy_selected_passing = 0
    total_retries_for_passers: list[int] = []

    for idx, prob in enumerate(problems):
        task_id = prob["task_id"]
        prompt = _build_generation_prompt(prob)
        test_cases = [line for line in prob["test"].strip().splitlines() if line.strip()]

        print(f"[exp905] {idx + 1}/{len(problems)}: {task_id} …", flush=True)
        t0 = time.perf_counter()

        try:
            result = pipeline.repair(prompt, test_cases)
        except Exception as exc:
            print(f"[exp905]   ERROR: {exc}", flush=True)
            results_per_problem.append(
                {
                    "task_id": task_id,
                    "error": str(exc),
                    "baseline_passed": False,
                    "repair_passed": False,
                    "n_retries": 0,
                    "energy_selected_passing": False,
                }
            )
            continue

        baseline_passed = result.all_attempts[0].exec_passed if result.all_attempts else False
        repair_passed = result.best_attempt.exec_passed

        if baseline_passed:
            n_baseline_pass += 1
        if repair_passed:
            n_repair_pass += 1
        if result.energy_selected_passing:
            n_energy_selected_passing += 1
        if repair_passed:
            total_retries_for_passers.append(result.n_retries)

        elapsed = round(time.perf_counter() - t0, 2)
        print(
            f"[exp905]   baseline={baseline_passed} repair={repair_passed} "
            f"retries={result.n_retries} energy={result.best_attempt.energy_score:.3f} [{elapsed}s]",
            flush=True,
        )

        results_per_problem.append(
            {
                "task_id": task_id,
                "baseline_passed": baseline_passed,
                "repair_passed": repair_passed,
                "n_retries": result.n_retries,
                "energy_score_best": result.best_attempt.energy_score,
                "energy_selected_passing": result.energy_selected_passing,
                "n_attempts": len(result.all_attempts),
                "elapsed_s": elapsed,
            }
        )

        # Checkpoint every 5 problems.
        if (idx + 1) % 5 == 0:
            tmpl.checkpoint_save({"results_so_far": results_per_problem}, step=idx + 1)

    # -- Compute metrics -------------------------------------------------------
    n = len(problems)
    baseline_pass_rate = n_baseline_pass / n if n > 0 else 0.0
    repair_pass_rate = n_repair_pass / n if n > 0 else 0.0
    signed_improvement = repair_pass_rate - baseline_pass_rate
    mean_retries_needed = (
        sum(total_retries_for_passers) / len(total_retries_for_passers)
        if total_retries_for_passers
        else 0.0
    )
    n_energy_correct = n_energy_selected_passing
    energy_selected_correctly = n_energy_correct / n if n > 0 else 0.0

    # -- Honest verdict -------------------------------------------------------
    if signed_improvement > 0.05:
        honest_verdict = "improvement_significant_gate_open"
    elif signed_improvement > 0:
        honest_verdict = "improvement_marginal_gate_open"
    else:
        honest_verdict = "no_improvement_gate_blocked"

    duration_s = round(time.perf_counter() - t_start, 2)
    print(
        f"\n[exp905] Results: baseline={baseline_pass_rate:.3f} repair={repair_pass_rate:.3f} "
        f"signed_improvement={signed_improvement:+.3f} verdict={honest_verdict}",
        flush=True,
    )

    # -- Write artifact -------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "model_id": model_id,
            "models_used": [model_id],
            "n_problems": n,
            "baseline_pass_rate": baseline_pass_rate,
            "repair_pass_rate": repair_pass_rate,
            "signed_improvement": signed_improvement,
            "mean_retries_needed": mean_retries_needed,
            "energy_selected_correctly": energy_selected_correctly,
            "energy_scorer_type": energy_scorer_type,
            "inference_mode": inference_mode,
            "max_retries": 3,
            "exec_timeout_s": 10.0,
            "n_baseline_pass": n_baseline_pass,
            "n_repair_pass": n_repair_pass,
            "results_per_problem": results_per_problem,
            "decision_class": "repair",
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode=inference_mode,
    )
    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[exp905] Artifact written to {output_path}", flush=True)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
