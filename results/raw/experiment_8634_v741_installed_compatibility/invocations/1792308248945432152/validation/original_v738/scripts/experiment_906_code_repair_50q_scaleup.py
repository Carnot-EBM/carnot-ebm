#!/usr/bin/env python3
"""Experiment 906: IterativeSelfRepair Scale-Up — 50 HumanEval, Qwen + Gemma cross-model.

**Researcher summary:**
    Exp 905 confirmed IterativeSelfRepair works: signed_improvement=+0.68 on 25 HumanEval
    problems with Gemma4-E4B-it (transformers fallback path).  This experiment scales to 50
    questions and adds the second SOTA model (unsloth/gemma-4-31B-it-GGUF) alongside the
    primary SOTA model (unsloth/Qwen3.6-35B-A3B-GGUF) to answer the cross-model question:
    does the Ising energy scorer correctly select the better model/attempt when both run?

**Gate:**
    Reads results/experiment_905_iterative_self_repair_v1.json.  Proceeds only if
    signed_improvement > 0.  If not, writes a blocked artifact and exits.

**Models:**
    1. unsloth/Qwen3.6-35B-A3B-GGUF  (llama.cpp, ~3B active params, primary SOTA)
    2. unsloth/gemma-4-31B-it-GGUF   (llama.cpp, 31B dense, second SOTA)
    Fallback for both: google/gemma-4-E4B-it via transformers (tiny model, CPU-capable).
    When both SOTA GGUFs are absent we still produce a valid artifact but label it
    'fallback_model_used' so the retrospective knows headline quality is degraded.

**Cross-model energy selection:**
    For each problem we run both models through IterativeSelfRepair(max_retries=3).
    The energy scorer returns a scalar per attempt.  We compare the best-attempt energy
    score from each model and declare the energy scorer "correct" when the model it
    selected actually has a higher pass rate (or equals the other model).
    Formally: energy_picked_correct if the lower-energy model's repair_passed >= the
    other model's repair_passed for that question.

**Metrics:**
    - qwen_repair_pass_rate / qwen_signed_improvement
    - gemma_repair_pass_rate / gemma_signed_improvement
    - cross_model_energy_selection_accuracy: fraction of questions where energy picked
      the correct (or tied) model
    - combined_pass_rate: fraction where AT LEAST ONE model passed after repair

**Honest-verdict mapping:**
    "strong_improvement_code_repair_milestone_achieved"  — max(signed_improvement) > 0.1
    "improvement_confirmed_scale_up"                     — max > 0
    "no_improvement_investigate"                         — max <= 0

Spec: REQ-CODE-033 (IterativeSelfRepair pipeline),
      SCENARIO-CODE-031 (retry with execution feedback),
      REQ-REPAIR-022 (SOTA GGUF required for headline results)
"""

from __future__ import annotations

import json
import os
import sys
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

_DELIVERABLE = "results/experiment_906_code_repair_50q_scaleup.json"
_GATE_RESULT = "results/experiment_905_iterative_self_repair_v1.json"

# ---------------------------------------------------------------------------
# 50 HumanEval problems (inline fallback when datasets unavailable)
# Problems 0-24 match Exp 905; problems 25-49 extend the benchmark.
# ---------------------------------------------------------------------------

_INLINE_PROBLEMS: list[dict[str, Any]] = [
    # ---- HumanEval/0-24: same as Exp 905 ----
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
    # ---- HumanEval/25-49: scale-up problems ----
    {
        "task_id": "HumanEval/25",
        "prompt": (
            "from typing import List\n\n"
            "def factorize(n: int) -> List[int]:\n"
            '    """ Return list of prime factors of given integer in the order from smallest to largest.\n'
            "    Each of the factors should be listed number of times corresponding to how many times\n"
            "    it appeares in factorization.\n"
            "    Input number should be equal to the product of all factors\n"
            "    >>> factorize(8)\n"
            "    [2, 2, 2]\n"
            "    >>> factorize(25)\n"
            "    [5, 5]\n"
            "    >>> factorize(70)\n"
            "    [2, 5, 7]\n"
            '    """\n'
        ),
        "entry_point": "factorize",
        "test": (
            "assert factorize(8) == [2, 2, 2]\n"
            "assert factorize(25) == [5, 5]\n"
            "assert factorize(70) == [2, 5, 7]\n"
            "assert factorize(1) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/26",
        "prompt": (
            "from typing import List\n\n"
            "def remove_duplicates(numbers: List[int]) -> List[int]:\n"
            '    """ From a list of integers, remove all elements that occur more than once.\n'
            "    Keep order of elements left the same as in the input.\n"
            "    >>> remove_duplicates([1, 2, 3, 2, 4])\n"
            "    [1, 3, 4]\n"
            '    """\n'
        ),
        "entry_point": "remove_duplicates",
        "test": (
            "assert remove_duplicates([]) == []\n"
            "assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]\n"
            "assert remove_duplicates([1, 2, 3, 4]) == [1, 2, 3, 4]\n"
        ),
    },
    {
        "task_id": "HumanEval/27",
        "prompt": (
            "def flip_case(string: str) -> str:\n"
            '    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.\n'
            "    >>> flip_case('Hello')\n"
            "    'hELLO'\n"
            '    """\n'
        ),
        "entry_point": "flip_case",
        "test": (
            "assert flip_case('') == ''\n"
            "assert flip_case('Hello') == 'hELLO'\n"
            "assert flip_case('These violent delights have violent ends') == 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'\n"
        ),
    },
    {
        "task_id": "HumanEval/28",
        "prompt": (
            "from typing import List\n\n"
            "def concatenate(strings: List[str]) -> str:\n"
            '    """ Concatenate list of strings into a single string\n'
            "    >>> concatenate([])\n"
            "    ''\n"
            "    >>> concatenate(['a', 'b', 'c'])\n"
            "    'abc'\n"
            '    """\n'
        ),
        "entry_point": "concatenate",
        "test": (
            "assert concatenate([]) == ''\n"
            "assert concatenate(['a', 'b', 'c']) == 'abc'\n"
            "assert concatenate(['hello', ' ', 'world']) == 'hello world'\n"
        ),
    },
    {
        "task_id": "HumanEval/29",
        "prompt": (
            "from typing import List\n\n"
            "def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:\n"
            '    """ Filter an input list of strings only for ones that start with a given prefix.\n'
            "    >>> filter_by_prefix([], 'a')\n"
            "    []\n"
            "    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')\n"
            "    ['abc', 'array']\n"
            '    """\n'
        ),
        "entry_point": "filter_by_prefix",
        "test": (
            "assert filter_by_prefix([], 'a') == []\n"
            "assert filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a') == ['abc', 'array']\n"
        ),
    },
    {
        "task_id": "HumanEval/30",
        "prompt": (
            "def get_positive(l: list) -> list:\n"
            '    """Return only positive numbers in the list.\n'
            "    >>> get_positive([-1, 2, -4, 3, 5])\n"
            "    [2, 3, 5]\n"
            "    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n"
            "    [5, 3, 2, 3, 9, 123, 1]\n"
            '    """\n'
        ),
        "entry_point": "get_positive",
        "test": (
            "assert get_positive([-1, 2, -4, 3, 5]) == [2, 3, 5]\n"
            "assert get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 9, 123, 1]\n"
            "assert get_positive([]) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/31",
        "prompt": (
            "def is_prime(n: int) -> bool:\n"
            '    """Return true if a given number is prime, and false otherwise.\n'
            "    >>> is_prime(6)\n"
            "    False\n"
            "    >>> is_prime(101)\n"
            "    True\n"
            "    >>> is_prime(11)\n"
            "    True\n"
            "    >>> is_prime(13441)\n"
            "    True\n"
            "    >>> is_prime(61)\n"
            "    True\n"
            "    >>> is_prime(4)\n"
            "    False\n"
            "    >>> is_prime(1)\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "is_prime",
        "test": (
            "assert is_prime(6) == False\n"
            "assert is_prime(101) == True\n"
            "assert is_prime(11) == True\n"
            "assert is_prime(13441) == True\n"
            "assert is_prime(61) == True\n"
            "assert is_prime(4) == False\n"
            "assert is_prime(1) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/32",
        "prompt": (
            "import math\n\n"
            "def poly(xs: list, x: float):\n"
            '    """\n'
            "    Evaluates polynomial with coefficients xs at point x.\n"
            "    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n\n"
            '    """\n'
            "    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n\n\n"
            "def find_zero(xs: list):\n"
            '    """ xs are coefficients of a polynomial.\n'
            "    find_zero find x such that poly(x) = 0.\n"
            "    find_zero returns only only zero point, even if there are many.\n"
            "    Moreover, find_zero only takes list xs having even number of coefficients\n"
            "    and largest non zero coefficient as it guarantees\n"
            "    a solution.\n"
            "    >>> round(find_zero([1, 2]), 2)\n"
            "    -0.5\n"
            "    >>> round(find_zero([-6, 11, -6, 1]), 2)\n"
            "    1.0\n"
            '    """\n'
        ),
        "entry_point": "find_zero",
        "test": (
            "import math\n"
            "def poly(xs, x): return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n"
            "assert round(find_zero([1, 2]), 2) == -0.5\n"
            "assert round(find_zero([-6, 11, -6, 1]), 2) == 1.0\n"
        ),
    },
    {
        "task_id": "HumanEval/33",
        "prompt": (
            "def sort_third(l: list) -> list:\n"
            '    """This function takes a list l and returns a list l\' such that\n'
            "    l' is identical to l in the indicies that are not divisible by three, while its values at the indicies that are\n"
            "    divisible by three are equal to the values of the corresponding indicies of l, but sorted.\n"
            "    >>> sort_third([1, 2, 3])\n"
            "    [1, 2, 3]\n"
            "    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n"
            "    [2, 6, 3, 4, 8, 9, 5]\n"
            '    """\n'
        ),
        "entry_point": "sort_third",
        "test": (
            "assert sort_third([1, 2, 3]) == [1, 2, 3]\n"
            "assert sort_third([5, 6, 3, 4, 8, 9, 2]) == [2, 6, 3, 4, 8, 9, 5]\n"
        ),
    },
    {
        "task_id": "HumanEval/34",
        "prompt": (
            "def unique(l: list) -> list:\n"
            '    """Return sorted unique elements in a list\n'
            "    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])\n"
            "    [0, 2, 3, 5, 9, 123]\n"
            '    """\n'
        ),
        "entry_point": "unique",
        "test": (
            "assert unique([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [0, 2, 3, 5, 9, 123]\n"
            "assert unique([]) == []\n"
            "assert unique([1]) == [1]\n"
        ),
    },
    {
        "task_id": "HumanEval/35",
        "prompt": (
            "def max_element(l: list) -> int:\n"
            '    """Return maximum element in the list.\n'
            "    >>> max_element([1, 2, 3])\n"
            "    3\n"
            "    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n"
            "    123\n"
            '    """\n'
        ),
        "entry_point": "max_element",
        "test": (
            "assert max_element([1, 2, 3]) == 3\n"
            "assert max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == 123\n"
        ),
    },
    {
        "task_id": "HumanEval/36",
        "prompt": (
            "def fizz_buzz(n: int) -> int:\n"
            '    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.\n'
            "    >>> fizz_buzz(50)\n"
            "    0\n"
            "    >>> fizz_buzz(78)\n"
            "    2\n"
            "    >>> fizz_buzz(79)\n"
            "    3\n"
            '    """\n'
        ),
        "entry_point": "fizz_buzz",
        "test": (
            "assert fizz_buzz(50) == 0\nassert fizz_buzz(78) == 2\nassert fizz_buzz(79) == 3\n"
        ),
    },
    {
        "task_id": "HumanEval/37",
        "prompt": (
            "def sort_even(l: list) -> list:\n"
            '    """This function takes a list l and returns a list l\' such that\n'
            "    l' is identical to l in the odd indicies, while its values at the even indicies are equal\n"
            "    to the values of the even indicies of l, but sorted.\n"
            "    >>> sort_even([1, 2, 3])\n"
            "    [1, 2, 3]\n"
            "    >>> sort_even([5, 6, 3, 4])\n"
            "    [3, 6, 5, 4]\n"
            '    """\n'
        ),
        "entry_point": "sort_even",
        "test": (
            "assert sort_even([1, 2, 3]) == [1, 2, 3]\n"
            "assert sort_even([5, 6, 3, 4]) == [3, 6, 5, 4]\n"
        ),
    },
    {
        "task_id": "HumanEval/38",
        "prompt": (
            "def encode_cyclic(s: str):\n"
            '    """\n'
            "    returns encoded string by cycling groups of three characters.\n"
            '    """\n'
            "    # split string to groups. Each of length 3.\n"
            "    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n"
            "    # cycle elements in each group. Unless group has fewer elements than 3.\n"
            "    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n"
            "    return ''.join(groups)\n\n\n"
            "def decode_cyclic(s: str):\n"
            '    """\n'
            "    takes as input string encoded with encode_cyclic function. Returns decoded string.\n"
            '    """\n'
        ),
        "entry_point": "decode_cyclic",
        "test": (
            "def encode_cyclic(s):\n"
            "    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n"
            "    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n"
            "    return ''.join(groups)\n"
            "assert decode_cyclic(encode_cyclic('abc')) == 'abc'\n"
            "assert decode_cyclic(encode_cyclic('hello world')) == 'hello world'\n"
            "assert decode_cyclic(encode_cyclic('')) == ''\n"
        ),
    },
    {
        "task_id": "HumanEval/39",
        "prompt": (
            "def prime_fib(n: int) -> int:\n"
            '    """\n'
            "    prime_fib returns n-th number that is a Fibonacci number and it's also prime.\n"
            "    >>> prime_fib(1)\n"
            "    2\n"
            "    >>> prime_fib(2)\n"
            "    3\n"
            "    >>> prime_fib(3)\n"
            "    5\n"
            "    >>> prime_fib(4)\n"
            "    13\n"
            "    >>> prime_fib(5)\n"
            "    89\n"
            '    """\n'
        ),
        "entry_point": "prime_fib",
        "test": (
            "assert prime_fib(1) == 2\n"
            "assert prime_fib(2) == 3\n"
            "assert prime_fib(3) == 5\n"
            "assert prime_fib(4) == 13\n"
            "assert prime_fib(5) == 89\n"
        ),
    },
    {
        "task_id": "HumanEval/40",
        "prompt": (
            "def triples_sum_to_zero(l: list) -> bool:\n"
            '    """\n'
            "    triples_sum_to_zero takes a list of integers as an input.\n"
            "    it returns True if there are three distinct elements in the list that\n"
            "    sum to zero, and False otherwise.\n"
            "    >>> triples_sum_to_zero([1, 3, 5, 0])\n"
            "    False\n"
            "    >>> triples_sum_to_zero([1, 3, -2, 1])\n"
            "    True\n"
            "    >>> triples_sum_to_zero([1, 2, 3, 7])\n"
            "    False\n"
            "    >>> triples_sum_to_zero([2, 4, -5, 3, 9, 7])\n"
            "    True\n"
            "    >>> triples_sum_to_zero([1])\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "triples_sum_to_zero",
        "test": (
            "assert triples_sum_to_zero([1, 3, 5, 0]) == False\n"
            "assert triples_sum_to_zero([1, 3, -2, 1]) == True\n"
            "assert triples_sum_to_zero([1, 2, 3, 7]) == False\n"
            "assert triples_sum_to_zero([2, 4, -5, 3, 9, 7]) == True\n"
            "assert triples_sum_to_zero([1]) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/41",
        "prompt": (
            "def car_race_collision(n: int) -> int:\n"
            '    """\n'
            "    Imagine a road that's a perfectly straight infinitely long line.\n"
            "    n cars are driving left to right;  simultaneously, a different set of n cars\n"
            "    are driving right to left.   The two sets of cars start out being very far from\n"
            "    each other.  All cars move in the same speed.  Two cars are said to collide\n"
            "    when a car that's moving left to right hits a car that's moving right to left.\n"
            "    However, the cars are infinitely sturdy and strong; as a result, they continue moving\n"
            "    in their trajectory as if they did not collide.\n\n"
            "    This function outputs the number of such collisions.\n"
            '    """\n'
        ),
        "entry_point": "car_race_collision",
        "test": (
            "assert car_race_collision(2) == 4\n"
            "assert car_race_collision(3) == 9\n"
            "assert car_race_collision(4) == 16\n"
            "assert car_race_collision(8) == 64\n"
        ),
    },
    {
        "task_id": "HumanEval/42",
        "prompt": (
            "def incr_list(l: list) -> list:\n"
            '    """Return list with elements incremented by 1.\n'
            "    >>> incr_list([1, 2, 3])\n"
            "    [2, 3, 4]\n"
            "    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])\n"
            "    [6, 4, 6, 3, 4, 4, 10, 1, 124]\n"
            '    """\n'
        ),
        "entry_point": "incr_list",
        "test": (
            "assert incr_list([1, 2, 3]) == [2, 3, 4]\n"
            "assert incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [6, 4, 6, 3, 4, 4, 10, 1, 124]\n"
            "assert incr_list([]) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/43",
        "prompt": (
            "def pairs_sum_to_zero(l: list) -> bool:\n"
            '    """\n'
            "    pairs_sum_to_zero takes a list of integers as an input.\n"
            "    it returns True if there are two distinct elements in the list that\n"
            "    sum to zero, and False otherwise.\n"
            "    >>> pairs_sum_to_zero([1, 3, 5, 0])\n"
            "    False\n"
            "    >>> pairs_sum_to_zero([1, 3, -2, 1])\n"
            "    False\n"
            "    >>> pairs_sum_to_zero([1, 2, 3, 7])\n"
            "    False\n"
            "    >>> pairs_sum_to_zero([2, 4, -5, 3, 9, 7])\n"
            "    False\n"
            "    >>> pairs_sum_to_zero([1])\n"
            "    False\n"
            "    >>> pairs_sum_to_zero([-1, 1])\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "pairs_sum_to_zero",
        "test": (
            "assert pairs_sum_to_zero([1, 3, 5, 0]) == False\n"
            "assert pairs_sum_to_zero([1, 3, -2, 1]) == False\n"
            "assert pairs_sum_to_zero([-1, 1]) == True\n"
            "assert pairs_sum_to_zero([-1, -1]) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/44",
        "prompt": (
            "def change_base(x: int, base: int) -> str:\n"
            '    """Change numerical base of input number x to base.\n'
            "    return string representation after the conversion.\n"
            "    base numbers are less than 10.\n"
            "    >>> change_base(8, 3)\n"
            "    '22'\n"
            "    >>> change_base(8, 2)\n"
            "    '1000'\n"
            "    >>> change_base(7, 2)\n"
            "    '111'\n"
            '    """\n'
        ),
        "entry_point": "change_base",
        "test": (
            "assert change_base(8, 3) == '22'\n"
            "assert change_base(8, 2) == '1000'\n"
            "assert change_base(7, 2) == '111'\n"
            "assert change_base(10, 10) == '10'\n"
        ),
    },
    {
        "task_id": "HumanEval/45",
        "prompt": (
            "def triangle_area(a: int, h: int) -> float:\n"
            '    """Given length of a side and high return area for a triangle.\n'
            "    >>> triangle_area(5, 3)\n"
            "    7.5\n"
            '    """\n'
        ),
        "entry_point": "triangle_area",
        "test": (
            "assert triangle_area(5, 3) == 7.5\n"
            "assert triangle_area(2, 2) == 2.0\n"
            "assert triangle_area(10, 8) == 40.0\n"
        ),
    },
    {
        "task_id": "HumanEval/46",
        "prompt": (
            "def fib4(n: int) -> int:\n"
            '    """The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that\'s defined as follows:\n'
            "    fib4(0) -> 0\n"
            "    fib4(1) -> 0\n"
            "    fib4(2) -> 2\n"
            "    fib4(3) -> 0\n"
            "    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).\n"
            "    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion.\n"
            "    >>> fib4(5)\n"
            "    4\n"
            "    >>> fib4(6)\n"
            "    8\n"
            "    >>> fib4(7)\n"
            "    14\n"
            '    """\n'
        ),
        "entry_point": "fib4",
        "test": (
            "assert fib4(5) == 4\n"
            "assert fib4(6) == 8\n"
            "assert fib4(7) == 14\n"
            "assert fib4(0) == 0\n"
            "assert fib4(1) == 0\n"
        ),
    },
    {
        "task_id": "HumanEval/47",
        "prompt": (
            "def median(l: list) -> float:\n"
            '    """Return median of elements in the list l.\n'
            "    >>> median([3, 1, 2, 4, 5])\n"
            "    3\n"
            "    >>> median([-10, 4, 6, 1000, 10, 20])\n"
            "    15.0\n"
            '    """\n'
        ),
        "entry_point": "median",
        "test": (
            "assert median([3, 1, 2, 4, 5]) == 3\n"
            "assert median([-10, 4, 6, 1000, 10, 20]) == 15.0\n"
        ),
    },
    {
        "task_id": "HumanEval/48",
        "prompt": (
            "def is_palindrome(text: str) -> bool:\n"
            '    """\n'
            "    Checks if given string is a palindrome\n"
            "    >>> is_palindrome('')\n"
            "    True\n"
            "    >>> is_palindrome('aba')\n"
            "    True\n"
            "    >>> is_palindrome('aaaaa')\n"
            "    True\n"
            "    >>> is_palindrome('zbcd')\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "is_palindrome",
        "test": (
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('aba') == True\n"
            "assert is_palindrome('aaaaa') == True\n"
            "assert is_palindrome('zbcd') == False\n"
        ),
    },
    {
        "task_id": "HumanEval/49",
        "prompt": (
            "def modp(n: int, p: int) -> int:\n"
            '    """Return 2^n modulo p (be aware of numerics).\n'
            "    >>> modp(3, 5)\n"
            "    3\n"
            "    >>> modp(1101, 101)\n"
            "    2\n"
            "    >>> modp(0, 101)\n"
            "    1\n"
            "    >>> modp(3, 11)\n"
            "    8\n"
            "    >>> modp(100, 101)\n"
            "    1\n"
            '    """\n'
        ),
        "entry_point": "modp",
        "test": (
            "assert modp(3, 5) == 3\n"
            "assert modp(1101, 101) == 2\n"
            "assert modp(0, 101) == 1\n"
            "assert modp(3, 11) == 8\n"
            "assert modp(100, 101) == 1\n"
        ),
    },
]

assert len(_INLINE_PROBLEMS) == 50, f"Expected 50 inline problems, got {len(_INLINE_PROBLEMS)}"


# ---------------------------------------------------------------------------
# Energy scorer
# ---------------------------------------------------------------------------


class _TokenLengthEnergyScorer:
    """Fallback energy scorer: shorter responses get lower energy.

    Used only when the Ising model is unavailable (e.g., JAX not installed).
    Headline claims must use ising_model, not this heuristic.
    """

    def score(self, text: str) -> float:
        """Return word count as a proxy energy (shorter code = lower energy = better)."""
        return float(len(text.split()))


def _build_energy_scorer() -> tuple[Any, str]:
    """Try to load the Ising energy scorer; fall back to token-length heuristic."""
    try:
        from carnot.models.ising import IsingConfig, IsingModel  # noqa: PLC0415
        import jax.random as jrandom  # noqa: PLC0415

        config = IsingConfig(input_dim=64, coupling_init="xavier_uniform")
        model = IsingModel(config, key=jrandom.PRNGKey(906))

        class _IsingScorer:
            def __init__(self, m: Any) -> None:
                self._m = m

            def score(self, text: str) -> float:
                import jax.numpy as jnp  # noqa: PLC0415

                chars = [ord(c) % 2 * 2 - 1 for c in text[:64]]
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

    Loads a GGUF model file through llama.cpp.  This is the SOTA inference
    path mandated for headline results (CLAUDE.md).
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate a completion and return only the assistant reply text."""
        output = self._model(
            prompt,
            max_tokens=512,
            temperature=0.0,
            echo=False,
        )
        return output["choices"][0]["text"].strip()


class _TransformersRunner:
    """LLM runner backed by HuggingFace transformers.

    Used as a fallback when llama.cpp is not available or no GGUF is cached.
    Results from this path are labelled 'fallback_model_used' in the artifact.
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
# Model loading — try GGUF path first, then transformers fallback
# ---------------------------------------------------------------------------


def _try_load_gguf(hf_id: str, label: str) -> tuple[Any, str] | None:
    """Try to load a GGUF model via llama.cpp from the HF cache.

    Returns (runner, model_id) on success, None if the model is not cached
    or llama.cpp is unavailable.  Does not raise — failures are silent so
    the experiment can continue with a fallback.
    """
    try:
        from llama_cpp import Llama  # noqa: PLC0415
        from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

        specs = cached_sota_pair(gpu_indices=(0,))
        if specs is None:
            return None
        for spec in specs:
            if hf_id in spec.get("hf_id", ""):
                model_path = spec.get("model_path")
                if model_path and Path(model_path).exists():
                    print(f"[exp906] Loading GGUF {hf_id} from {model_path} …", flush=True)
                    llm = Llama(
                        model_path=str(model_path),
                        n_gpu_layers=-1,
                        n_ctx=4096,
                        verbose=False,
                    )
                    return _LlamaCppRunner(llm), hf_id
        return None
    except Exception as exc:
        print(f"[exp906] GGUF load for {hf_id} failed: {exc}", flush=True)
        return None


def _load_transformers_fallback() -> tuple[Any, str]:
    """Load the tiny Gemma4-E4B fallback model via transformers.

    This is the CPU-capable path.  Results from this path are valid for
    pipeline structure testing but NOT for headline benchmark claims.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_id = "google/gemma-4-E4B-it"
    print(f"[exp906] Loading transformers fallback: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return _TransformersRunner(model, tokenizer), model_id


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------


def _build_generation_prompt(problem: dict[str, Any]) -> str:
    """Build the initial generation prompt for a HumanEval problem.

    Keeps the prompt minimal — function signature + docstring + one-line
    instruction — consistent with Exp 905 for fair cross-experiment comparison.
    """
    return (
        "Complete the following Python function. "
        "Output ONLY the function body code (no imports, no prose):\n\n" + problem["prompt"]
    )


# ---------------------------------------------------------------------------
# Per-model repair runner
# ---------------------------------------------------------------------------


def _run_model_on_problems(
    runner: Any,
    model_id: str,
    problems: list[dict[str, Any]],
    energy_scorer: Any,
    tmpl: ExperimentTemplate,
    prefix: str,
) -> list[dict[str, Any]]:
    """Run IterativeSelfRepair for one model across all problems.

    Returns a list of per-problem result dicts with keys:
        task_id, baseline_passed, repair_passed, n_retries,
        energy_score_best, energy_selected_passing, n_attempts, elapsed_s
    """
    from carnot.pipeline.iterative_self_repair import IterativeSelfRepair  # noqa: PLC0415

    pipeline = IterativeSelfRepair(
        llm_runner=runner,
        energy_scorer=energy_scorer,
        max_retries=3,
        sandbox=False,
        exec_timeout_s=10.0,
    )

    results: list[dict[str, Any]] = []
    for idx, prob in enumerate(problems):
        task_id = prob["task_id"]
        prompt = _build_generation_prompt(prob)
        test_cases = [line for line in prob["test"].strip().splitlines() if line.strip()]

        print(f"[exp906/{prefix}] {idx + 1}/{len(problems)}: {task_id} …", flush=True)
        t0 = time.perf_counter()

        try:
            result = pipeline.repair(prompt, test_cases)
        except Exception as exc:
            print(f"[exp906/{prefix}]   ERROR: {exc}", flush=True)
            results.append(
                {
                    "task_id": task_id,
                    "error": str(exc),
                    "baseline_passed": False,
                    "repair_passed": False,
                    "n_retries": 0,
                    "energy_score_best": 0.0,
                    "energy_selected_passing": False,
                    "n_attempts": 0,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
            )
            continue

        baseline_passed = result.all_attempts[0].exec_passed if result.all_attempts else False
        repair_passed = result.best_attempt.exec_passed
        elapsed = round(time.perf_counter() - t0, 2)

        print(
            f"[exp906/{prefix}]   baseline={baseline_passed} repair={repair_passed} "
            f"retries={result.n_retries} energy={result.best_attempt.energy_score:.3f} [{elapsed}s]",
            flush=True,
        )

        results.append(
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

        if (idx + 1) % 10 == 0:
            tmpl.checkpoint_save(
                {f"{prefix}_results_so_far": results},
                step=idx + 1,
            )

    return results


# ---------------------------------------------------------------------------
# Cross-model energy selection accuracy
# ---------------------------------------------------------------------------


def _compute_cross_model_accuracy(
    qwen_results: list[dict[str, Any]],
    gemma_results: list[dict[str, Any]],
) -> float:
    """Compute fraction of questions where the energy scorer picked the better model.

    For each question we compare the best-attempt energy score from each model.
    The energy scorer is "correct" when the model it selected (lower energy score)
    has repair_passed >= the other model's repair_passed for that question.
    When energy scores are equal we count it as correct (tie).

    This directly measures whether Carnot's energy objective can act as a
    model-selection oracle — if yes, callers can run both models and let energy
    choose, getting the benefit of the stronger model without knowing which it is.
    """
    n = min(len(qwen_results), len(gemma_results))
    if n == 0:
        return 0.0

    correct = 0
    for q, g in zip(qwen_results[:n], gemma_results[:n]):
        q_energy = q.get("energy_score_best", 0.0)
        g_energy = g.get("energy_score_best", 0.0)
        q_passed = q.get("repair_passed", False)
        g_passed = g.get("repair_passed", False)

        if q_energy <= g_energy:
            # Energy picked Qwen.  Correct if Qwen passed or both failed equally.
            if q_passed or not g_passed:
                correct += 1
        else:
            # Energy picked Gemma.  Correct if Gemma passed or both failed equally.
            if g_passed or not q_passed:
                correct += 1

    return correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate Exp 906: scale-up IterativeSelfRepair and write the deliverable."""
    # ---- Gate check --------------------------------------------------------
    gate_path = _REPO_ROOT / _GATE_RESULT
    try:
        gate_data = json.loads(gate_path.read_text())
        signed_improvement_905 = gate_data.get("signed_improvement", 0.0)
    except Exception as exc:
        signed_improvement_905 = 0.0
        print(f"[exp906] WARNING: Could not read gate result: {exc}", flush=True)

    if signed_improvement_905 <= 0:
        print(
            f"[exp906] GATE BLOCKED: Exp 905 signed_improvement={signed_improvement_905} <= 0. Skipping.",
            flush=True,
        )
        # Write a minimal blocked artifact so the conductor sees a deliverable.
        blocked = {
            "experiment": 906,
            "title": "IterativeSelfRepair Scale-Up — 50 HumanEval, Qwen + Gemma",
            "run_date": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .strftime("%Y%m%d"),
            "started_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": 0.0,
            "status": "skipped",
            "honest_verdict": "skipped_gate_blocked_exp905_no_improvement",
            "gate_blocked": True,
            "exp905_signed_improvement": signed_improvement_905,
            "schema": [
                "experiment",
                "title",
                "run_date",
                "started_at",
                "finished_at",
                "duration_s",
                "status",
                "honest_verdict",
            ],
        }
        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.write_text(json.dumps(blocked, indent=2))
        return

    print(
        f"[exp906] Gate PASSED: Exp 905 signed_improvement={signed_improvement_905:.3f}", flush=True
    )

    # ---- Setup -------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=906,
        title="IterativeSelfRepair Scale-Up — 50 HumanEval, Qwen + Gemma",
        deliverable=_DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    t_start = time.perf_counter()

    # ---- Energy scorer -----------------------------------------------------
    energy_scorer, energy_scorer_type = _build_energy_scorer()
    print(f"[exp906] Energy scorer: {energy_scorer_type}", flush=True)

    # ---- Load 50 HumanEval problems ----------------------------------------
    try:
        from human_eval.data import read_problems  # noqa: PLC0415

        all_problems = read_problems()
        task_ids = sorted(all_problems.keys())[:50]
        problems: list[dict[str, Any]] = [all_problems[tid] for tid in task_ids]
        print(f"[exp906] Using human_eval package for {len(problems)} problems.", flush=True)
    except ImportError:
        problems = _INLINE_PROBLEMS
        print(
            f"[exp906] human_eval not installed — using inline problems ({len(problems)}).",
            flush=True,
        )

    # ---- Load Qwen (primary SOTA) ------------------------------------------
    qwen_result = _try_load_gguf("Qwen3.6-35B-A3B-GGUF", "qwen")
    if qwen_result is not None:
        qwen_runner, qwen_model_id = qwen_result
        qwen_fallback = False
    else:
        print(
            "[exp906] Qwen GGUF not cached — using transformers fallback for Qwen slot.", flush=True
        )
        try:
            qwen_runner, qwen_model_id = _load_transformers_fallback()
            qwen_fallback = True
        except Exception as exc:
            print(f"[exp906] Both Qwen and fallback failed: {exc}", flush=True)
            artifact = tmpl.build_result(
                {"model_load_error": str(exc), "traceback": tb.format_exc()},
                status="blocked",
                honest_verdict="blocked_model_load_failure",
            )
            output_path = _REPO_ROOT / _DELIVERABLE
            output_path.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

    # ---- Run Qwen on all 50 problems ---------------------------------------
    print(
        f"[exp906] Running Qwen model ({qwen_model_id}) on {len(problems)} problems …", flush=True
    )
    qwen_results = _run_model_on_problems(
        qwen_runner, qwen_model_id, problems, energy_scorer, tmpl, prefix="qwen"
    )

    # ---- Load Gemma (second SOTA) — unload Qwen memory first if possible ---
    # We try to delete the Qwen runner to free VRAM before loading Gemma.
    # If Qwen was a transformers model, force a GC pass.
    try:
        del qwen_runner
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass

    gemma_result = _try_load_gguf("gemma-4-31B-it-GGUF", "gemma")
    if gemma_result is not None:
        gemma_runner, gemma_model_id = gemma_result
        gemma_fallback = False
    else:
        print(
            "[exp906] Gemma GGUF not cached — using transformers fallback for Gemma slot.",
            flush=True,
        )
        try:
            gemma_runner, gemma_model_id = _load_transformers_fallback()
            gemma_fallback = True
        except Exception as exc:
            print(f"[exp906] Both Gemma and fallback failed: {exc}", flush=True)
            artifact = tmpl.build_result(
                {"model_load_error": str(exc), "traceback": tb.format_exc()},
                status="blocked",
                honest_verdict="blocked_gemma_load_failure",
            )
            output_path = _REPO_ROOT / _DELIVERABLE
            output_path.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

    # ---- Run Gemma on all 50 problems ---------------------------------------
    print(
        f"[exp906] Running Gemma model ({gemma_model_id}) on {len(problems)} problems …", flush=True
    )
    gemma_results = _run_model_on_problems(
        gemma_runner, gemma_model_id, problems, energy_scorer, tmpl, prefix="gemma"
    )

    # ---- Compute per-model metrics -----------------------------------------
    n = len(problems)

    def _pass_rates(results: list[dict[str, Any]]) -> tuple[float, float, float]:
        """Return (baseline_pass_rate, repair_pass_rate, signed_improvement)."""
        n_base = sum(1 for r in results if r.get("baseline_passed", False))
        n_rep = sum(1 for r in results if r.get("repair_passed", False))
        base_rate = n_base / n if n > 0 else 0.0
        rep_rate = n_rep / n if n > 0 else 0.0
        return base_rate, rep_rate, rep_rate - base_rate

    qwen_base, qwen_repair, qwen_signed = _pass_rates(qwen_results)
    gemma_base, gemma_repair, gemma_signed = _pass_rates(gemma_results)

    # Combined pass rate: fraction where at least one model passed after repair.
    combined_pass = (
        sum(
            1
            for q, g in zip(qwen_results, gemma_results)
            if q.get("repair_passed", False) or g.get("repair_passed", False)
        )
        / n
        if n > 0
        else 0.0
    )

    cross_model_accuracy = _compute_cross_model_accuracy(qwen_results, gemma_results)

    # ---- Honest verdict ----------------------------------------------------
    max_signed = max(qwen_signed, gemma_signed)
    if max_signed > 0.1:
        honest_verdict = "strong_improvement_code_repair_milestone_achieved"
    elif max_signed > 0:
        honest_verdict = "improvement_confirmed_scale_up"
    else:
        honest_verdict = "no_improvement_investigate"

    duration_s = round(time.perf_counter() - t_start, 2)

    print(
        f"\n[exp906] Qwen: baseline={qwen_base:.3f} repair={qwen_repair:.3f} delta={qwen_signed:+.3f}",
        flush=True,
    )
    print(
        f"[exp906] Gemma: baseline={gemma_base:.3f} repair={gemma_repair:.3f} delta={gemma_signed:+.3f}",
        flush=True,
    )
    print(
        f"[exp906] cross_model_energy_selection_accuracy={cross_model_accuracy:.3f}  "
        f"combined_pass_rate={combined_pass:.3f}  verdict={honest_verdict}",
        flush=True,
    )

    # ---- Write artifact ----------------------------------------------------
    models_used = list({qwen_model_id, gemma_model_id})
    inference_mode = "live_gpu"
    if qwen_fallback and gemma_fallback:
        inference_mode = "fallback_transformers_only"
    elif qwen_fallback or gemma_fallback:
        inference_mode = "mixed_gguf_and_fallback"

    artifact = tmpl.build_result(
        {
            "n_problems": n,
            "models_used": models_used,
            "qwen_model_id": qwen_model_id,
            "gemma_model_id": gemma_model_id,
            "energy_scorer_type": energy_scorer_type,
            "max_retries": 3,
            "exec_timeout_s": 10.0,
            "qwen_baseline_pass_rate": qwen_base,
            "qwen_repair_pass_rate": qwen_repair,
            "qwen_signed_improvement": qwen_signed,
            "gemma_baseline_pass_rate": gemma_base,
            "gemma_repair_pass_rate": gemma_repair,
            "gemma_signed_improvement": gemma_signed,
            "cross_model_energy_selection_accuracy": cross_model_accuracy,
            "combined_pass_rate": combined_pass,
            "exp905_signed_improvement": signed_improvement_905,
            "inference_mode": inference_mode,
            "decision_class": "repair",
            "qwen_results_per_problem": qwen_results,
            "gemma_results_per_problem": gemma_results,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode=inference_mode,
    )

    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[exp906] Artifact written to {output_path}", flush=True)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
