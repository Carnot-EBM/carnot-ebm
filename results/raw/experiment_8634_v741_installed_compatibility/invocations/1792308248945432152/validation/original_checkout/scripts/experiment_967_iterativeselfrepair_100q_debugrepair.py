#!/usr/bin/env python3
"""Experiment 967: IterativeSelfRepair 100q + DebugRepair Hypothesis Step.

**Researcher summary:**
    Exp 959 was blocked in milestone .74 because prior_failures were missing from the
    roadmap YAML.  This retry provides complete prior_failures and is otherwise identical
    in scope.

    Exp 905 (25q code, +0.68 signed improvement) and Exp 906 (50q cross-model, strong
    improvement) validated IterativeSelfRepair on code.  This experiment scales to 100q
    and adds the DebugRepair hypothesis step from arXiv 2604.19305 (+8.2pp HumanEval).

**DebugRepair hypothesis step (arXiv 2604.19305):**
    Before generating a repair, the model is asked to generate a *hypothesis* about
    WHY the initial response was wrong.  The hypothesis is then fed into the repair
    prompt as explicit context:
        "My diagnosis of the error: [hypothesis]. Now generate a corrected response."

    The intuition: if the model first articulates a hypothesis about the bug, it is
    less likely to regenerate the same buggy code.  The hypothesis acts as an internal
    chain-of-thought step that names the root cause before the fix is written.

**Domains:**
    1. CODE — 100 HumanEval problems, 3 repair attempts each.
       - Measures code_repair_delta = repair_pass_rate - baseline_pass_rate
       - Measures hypothesis_contribution = pass_rate(with_hyp) - pass_rate(without_hyp)
       (we run a subset of 20 problems both ways to estimate contribution efficiently)
    2. MATH — 50 GSM8K problems.
       - No energy scoring; external feedback (correct answer Y/N) re-fed into repair.
       - Measures math_repair_delta = repair_pass_rate - baseline_pass_rate

**Prior failures addressed:**
    - Exp 959 blocked: missing prior_failures in roadmap YAML (not a real failure of
      the experiment logic itself; conductor refused to launch without the field).
    - No underlying algorithmic failure to address — the technique itself is untested
      at 100q scale; 905/906 results give us strong prior that it will work.

**Models:**
    Primary: unsloth/gemma-4-31B-it-GGUF  (llama.cpp, 31B dense, Gemma 4)
    Fallback: unsloth/Qwen3.6-35B-A3B-GGUF (llama.cpp, ~3B active, MoE)
    Last resort: google/gemma-4-E4B-it via transformers (CPU-capable tiny model)

**Honest verdict:**
    "iterative_repair_100q_viable" if EITHER domain shows delta > 0.
    "no_improvement_100q" otherwise.

**Output schema (required fields):**
    code_repair_delta: float
    math_repair_delta: float
    hypothesis_contribution: float
    n_problems_code: int
    n_problems_math: int
    model_used: str
    honest_verdict: str

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

_DELIVERABLE = "results/experiment_967_iterativeselfrepair_100q_debugrepair.json"

# ---------------------------------------------------------------------------
# 100 HumanEval problems (inline — extends Exp 906's 50 with problems 50-99)
# ---------------------------------------------------------------------------

# Reuse the first 50 from the Exp 906 inline set by importing dynamically.
# We extend with 50 more inline problems (HumanEval/50-99) below.

_CODE_PROBLEMS_50_99: list[dict[str, Any]] = [
    {
        "task_id": "HumanEval/50",
        "prompt": (
            "def encode_shift(s: str):\n"
            '    """\n'
            "    returns encoded string by shifting every character by 5 in the alphabet.\n"
            '    """\n'
            "    return ''.join(chr(((ord(ch) + 5 - ord('a')) % 26) + ord('a')) "
            "if ch.isalpha() and ch.islower() else ch for ch in s)\n\n\n"
            "def decode_shift(s: str):\n"
            '    """\n'
            "    takes as input string encoded with encode_shift function. Returns decoded string.\n"
            '    """\n'
        ),
        "entry_point": "decode_shift",
        "test": (
            "def encode_shift(s):\n"
            "    return ''.join(chr(((ord(ch) + 5 - ord('a')) % 26) + ord('a')) "
            "if ch.isalpha() and ch.islower() else ch for ch in s)\n"
            "assert decode_shift(encode_shift('hello')) == 'hello'\n"
            "assert decode_shift(encode_shift('abc')) == 'abc'\n"
            "assert decode_shift(encode_shift('')) == ''\n"
        ),
    },
    {
        "task_id": "HumanEval/51",
        "prompt": (
            "def remove_vowels(text):\n"
            '    """\n'
            "    remove_vowels is a function that takes string and returns string without vowels.\n"
            "    >>> remove_vowels('')\n"
            "    ''\n"
            "    >>> remove_vowels('abcdef')\n"
            "    'bcdf'\n"
            "    >>> remove_vowels('aaaaa')\n"
            "    ''\n"
            "    >>> remove_vowels('aaBAA')\n"
            "    'B'\n"
            "    >>> remove_vowels('zbcd')\n"
            "    'zbcd'\n"
            '    """\n'
        ),
        "entry_point": "remove_vowels",
        "test": (
            "assert remove_vowels('') == ''\n"
            "assert remove_vowels('abcdef') == 'bcdf'\n"
            "assert remove_vowels('aaaaa') == ''\n"
            "assert remove_vowels('aaBAA') == 'B'\n"
            "assert remove_vowels('zbcd') == 'zbcd'\n"
        ),
    },
    {
        "task_id": "HumanEval/52",
        "prompt": (
            "def below_threshold(l: list, t: int):\n"
            '    """Return True if all numbers in the list l are below threshold t.\n'
            "    >>> below_threshold([1, 2, 4, 10], 100)\n"
            "    True\n"
            "    >>> below_threshold([1, 20, 4, 10], 5)\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "below_threshold",
        "test": (
            "assert below_threshold([1, 2, 4, 10], 100) == True\n"
            "assert below_threshold([1, 20, 4, 10], 5) == False\n"
            "assert below_threshold([], 5) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/53",
        "prompt": (
            "def add(x: int, y: int):\n"
            '    """Add two numbers x and y\n'
            "    >>> add(2, 3)\n"
            "    5\n"
            "    >>> add(5, 7)\n"
            "    12\n"
            '    """\n'
        ),
        "entry_point": "add",
        "test": ("assert add(2, 3) == 5\nassert add(5, 7) == 12\nassert add(0, 0) == 0\n"),
    },
    {
        "task_id": "HumanEval/54",
        "prompt": (
            "def same_chars(s0: str, s1: str):\n"
            '    """\n'
            "    Check if two words have the same characters.\n"
            "    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')\n"
            "    True\n"
            "    >>> same_chars('abcd', 'dddddddabc')\n"
            "    True\n"
            "    >>> same_chars('dddddddabc', 'abcd')\n"
            "    True\n"
            "    >>> same_chars('eabcd', 'dddddddabc')\n"
            "    False\n"
            "    >>> same_chars('abcd', 'dddddddabce')\n"
            "    False\n"
            "    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabce')\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "same_chars",
        "test": (
            "assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True\n"
            "assert same_chars('abcd', 'dddddddabc') == True\n"
            "assert same_chars('eabcd', 'dddddddabc') == False\n"
            "assert same_chars('abcd', 'dddddddabce') == False\n"
        ),
    },
    {
        "task_id": "HumanEval/55",
        "prompt": (
            "def fib(n: int):\n"
            '    """Return n-th Fibonacci number.\n'
            "    >>> fib(10)\n"
            "    55\n"
            "    >>> fib(1)\n"
            "    1\n"
            "    >>> fib(8)\n"
            "    21\n"
            '    """\n'
        ),
        "entry_point": "fib",
        "test": (
            "assert fib(10) == 55\nassert fib(1) == 1\nassert fib(8) == 21\nassert fib(2) == 1\n"
        ),
    },
    {
        "task_id": "HumanEval/56",
        "prompt": (
            "def correct_bracketing(brackets: str):\n"
            '    """ brackets is a string of "<" and ">".\n'
            "    return True if every opening bracket has a corresponding closing bracket.\n\n"
            "    >>> correct_bracketing('<')\n"
            "    False\n"
            "    >>> correct_bracketing('<>')\n"
            "    True\n"
            "    >>> correct_bracketing('<<><>>')\n"
            "    True\n"
            "    >>> correct_bracketing('><<>')\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "correct_bracketing",
        "test": (
            "assert correct_bracketing('<') == False\n"
            "assert correct_bracketing('<>') == True\n"
            "assert correct_bracketing('<<><>>') == True\n"
            "assert correct_bracketing('><<>') == False\n"
            "assert correct_bracketing('') == True\n"
        ),
    },
    {
        "task_id": "HumanEval/57",
        "prompt": (
            "def monotonic(l: list):\n"
            '    """Return True is list elements are monotonically increasing or decreasing.\n'
            "    >>> monotonic([1, 2, 4, 20])\n"
            "    True\n"
            "    >>> monotonic([1, 20, 4, 10])\n"
            "    False\n"
            "    >>> monotonic([4, 1, 0, -10])\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "monotonic",
        "test": (
            "assert monotonic([1, 2, 4, 20]) == True\n"
            "assert monotonic([1, 20, 4, 10]) == False\n"
            "assert monotonic([4, 1, 0, -10]) == True\n"
            "assert monotonic([1, 1, 1]) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/58",
        "prompt": (
            "def common(l1: list, l2: list):\n"
            '    """Return sorted unique common elements for two lists.\n'
            "    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])\n"
            "    [1, 5, 653]\n"
            "    >>> common([5, 3, 2, 8], [3, 2])\n"
            "    [2, 3]\n\n"
            '    """\n'
        ),
        "entry_point": "common",
        "test": (
            "assert common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121]) == [1, 5, 653]\n"
            "assert common([5, 3, 2, 8], [3, 2]) == [2, 3]\n"
            "assert common([], []) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/59",
        "prompt": (
            "def largest_prime_factor(n: int):\n"
            '    """Return the largest prime factor of n. Assume n > 1 and is not a prime.\n'
            "    >>> largest_prime_factor(13195)\n"
            "    29\n"
            "    >>> largest_prime_factor(2048)\n"
            "    2\n"
            '    """\n'
        ),
        "entry_point": "largest_prime_factor",
        "test": (
            "assert largest_prime_factor(13195) == 29\n"
            "assert largest_prime_factor(2048) == 2\n"
            "assert largest_prime_factor(15) == 5\n"
        ),
    },
    {
        "task_id": "HumanEval/60",
        "prompt": (
            "def sum_to_n(n: int):\n"
            '    """sum_to_n is a function that sums numbers from 1 to n.\n'
            "    >>> sum_to_n(30)\n"
            "    465\n"
            "    >>> sum_to_n(100)\n"
            "    5050\n"
            "    >>> sum_to_n(5)\n"
            "    15\n"
            "    >>> sum_to_n(10)\n"
            "    55\n"
            "    >>> sum_to_n(1)\n"
            "    1\n"
            '    """\n'
        ),
        "entry_point": "sum_to_n",
        "test": (
            "assert sum_to_n(30) == 465\n"
            "assert sum_to_n(100) == 5050\n"
            "assert sum_to_n(5) == 15\n"
            "assert sum_to_n(1) == 1\n"
        ),
    },
    {
        "task_id": "HumanEval/61",
        "prompt": (
            "def correct_bracketing(brackets: str):\n"
            '    """ brackets is a string of "(" and ")".\n'
            "    return True if every opening bracket has a corresponding closing bracket.\n\n"
            "    >>> correct_bracketing('(')\n"
            "    False\n"
            "    >>> correct_bracketing('()')\n"
            "    True\n"
            "    >>> correct_bracketing('(()())')\n"
            "    True\n"
            "    >>> correct_bracketing(')(()')\n"
            "    False\n"
            '    """\n'
        ),
        "entry_point": "correct_bracketing",
        "test": (
            "assert correct_bracketing('(') == False\n"
            "assert correct_bracketing('()') == True\n"
            "assert correct_bracketing('(()())') == True\n"
            "assert correct_bracketing(')(()') == False\n"
        ),
    },
    {
        "task_id": "HumanEval/62",
        "prompt": (
            "def derivative(xs: list):\n"
            '    """ xs represent coefficients of a polynomial.\n'
            "    xs[0] + xs[1] * x + xs[2] * x^2 + ....\n"
            "    Return derivative of this polynomial in the same form.\n"
            "    >>> derivative([3, 1, 2, 4, 5])\n"
            "    [1, 4, 12, 20]\n"
            "    >>> derivative([1, 2, 3])\n"
            "    [2, 6]\n"
            '    """\n'
        ),
        "entry_point": "derivative",
        "test": (
            "assert derivative([3, 1, 2, 4, 5]) == [1, 4, 12, 20]\n"
            "assert derivative([1, 2, 3]) == [2, 6]\n"
            "assert derivative([5]) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/63",
        "prompt": (
            "def fibfib(n: int):\n"
            '    """The FibFib number sequence is a sequence similar to the Fibonacci sequnece that\'s\n'
            "    defined as follows:\n"
            "    fibfib(0) == 0\n"
            "    fibfib(1) == 0\n"
            "    fibfib(2) == 1\n"
            "    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).\n"
            "    Please write a function to efficiently compute the n-th element of the fibfib number\n"
            "    sequence.  Do not use recursion.\n"
            "    >>> fibfib(1)\n"
            "    0\n"
            "    >>> fibfib(5)\n"
            "    4\n"
            "    >>> fibfib(8)\n"
            "    24\n"
            '    """\n'
        ),
        "entry_point": "fibfib",
        "test": (
            "assert fibfib(1) == 0\n"
            "assert fibfib(5) == 4\n"
            "assert fibfib(8) == 24\n"
            "assert fibfib(0) == 0\n"
            "assert fibfib(2) == 1\n"
        ),
    },
    {
        "task_id": "HumanEval/64",
        "prompt": (
            'FIX = """\n'
            "Add more test cases.\n"
            '"""\n\n'
            "def vowels_count(s):\n"
            '    """Write a function vowels_count which takes a string representing\n'
            "    a word as input and returns the number of vowels in the string.\n"
            "    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a\n"
            "    vowel, but only when it is at the end of the given word.\n\n"
            "    Example:\n"
            "    >>> vowels_count('abcde')\n"
            "    2\n"
            "    >>> vowels_count('ACEDY')\n"
            "    3\n"
            '    """\n'
        ),
        "entry_point": "vowels_count",
        "test": (
            "assert vowels_count('abcde') == 2\n"
            "assert vowels_count('ACEDY') == 3\n"
            "assert vowels_count('éxamplë') == 2\n"
        ),
    },
    {
        "task_id": "HumanEval/65",
        "prompt": (
            "def circular_shift(x, shift):\n"
            '    """Circular shift the digits of the integer x, shift the digits right by shift\n'
            "    and return the result as a string.\n"
            "    If shift > number of digits, return digits reversed.\n"
            "    >>> circular_shift(12, 1)\n"
            "    '21'\n"
            "    >>> circular_shift(12, 2)\n"
            "    '12'\n"
            '    """\n'
        ),
        "entry_point": "circular_shift",
        "test": (
            "assert circular_shift(12, 1) == '21'\n"
            "assert circular_shift(12, 2) == '12'\n"
            "assert circular_shift(12, 3) == '21'\n"
            "assert circular_shift(1, 0) == '1'\n"
        ),
    },
    {
        "task_id": "HumanEval/66",
        "prompt": (
            "def digitSum(s):\n"
            '    """Task\n'
            "    Write a function that takes a string as input and returns the sum of the upper characters only'\n"
            "    ASCII codes.\n\n"
            "    Examples:\n"
            "    >>> digitSum('')\n"
            "    0\n"
            "    >>> digitSum('abAB')\n"
            "    131\n"
            "    >>> digitSum('abcCd')\n"
            "    67\n"
            "    >>> digitSum('helloE')\n"
            "    69\n"
            "    >>> digitSum('woArBld')\n"
            "    131\n"
            "    >>> digitSum('aAaaaXa')\n"
            "    153\n"
            '    """\n'
        ),
        "entry_point": "digitSum",
        "test": (
            "assert digitSum('') == 0\n"
            "assert digitSum('abAB') == 131\n"
            "assert digitSum('abcCd') == 67\n"
            "assert digitSum('helloE') == 69\n"
        ),
    },
    {
        "task_id": "HumanEval/67",
        "prompt": (
            "def fruit_distribution(s, n):\n"
            '    """\n'
            "    In this task, you will be given a string that represents a number of apples and oranges\n"
            "    that are distributed in a basket of fruit this basket contains\n"
            "    apples, oranges, and mango fruits. Given the string that represents the total number of\n"
            "    the oranges and apples and an integer that represent the total number of the fruits\n"
            "    in the basket return the number of the mango fruits in the basket.\n"
            "    for examble:\n"
            "    fruit_distribution('5 apples and 6 oranges', 19) ->19 - 5 - 6 = 8\n"
            "    fruit_distribution('0 apples and 1 oranges',3) -> 3 - 0 - 1 = 2\n"
            "    fruit_distribution('2 apples and 3 oranges', 100) -> 100 - 2 - 3 = 95\n"
            "    fruit_distribution('100 apples and 1 oranges',120) -> 120 - 100 - 1 = 19\n"
            '    """\n'
        ),
        "entry_point": "fruit_distribution",
        "test": (
            "assert fruit_distribution('5 apples and 6 oranges', 19) == 8\n"
            "assert fruit_distribution('0 apples and 1 oranges', 3) == 2\n"
            "assert fruit_distribution('2 apples and 3 oranges', 100) == 95\n"
        ),
    },
    {
        "task_id": "HumanEval/68",
        "prompt": (
            "def pluck(arr):\n"
            '    """\n'
            '    "Given an array representing a branch of a tree that has non-negative integer nodes\n'
            "    your task is to pluck one of the nodes and return it.\n"
            "    The plucked node should be the node with the smallest even value.\n"
            "    If multiple nodes with the same smallest even value are found return the node that has smallest index.\n\n"
            "    The plucked node should be returned in a list, [ smalest_value, its index ],\n"
            "    If there are no even values or the given array is empty, return [].\n\n"
            "    Example 1:\n"
            "        Input: [4,2,3]\n"
            "        Output: [2, 1]\n"
            "        Explanation: 2 has the smallest even value, and 2 has the smallest index.\n\n"
            "    Example 2:\n"
            "        Input: [1,2,3]\n"
            "        Output: [2, 1]\n"
            "        Explanation: 2 has the smallest even value, and 2 has the smallest index.\n\n"
            "    Example 3:\n"
            "        Input: []\n"
            "        Output: []\n\n"
            "    Example 4:\n"
            "        Input: [5, 0, 3, 0, 4, 2]\n"
            "        Output: [0, 1]\n"
            "        Explanation: 0 is the smallest value, but  there are two zeros,\n"
            "                     so we will choose the first zero, which has the smallest index.\n\n"
            "    Constraints:\n"
            "        * 1 <= nodes.length <= 10000\n"
            "        * 0 <= node.value\n"
            '    """\n'
        ),
        "entry_point": "pluck",
        "test": (
            "assert pluck([4, 2, 3]) == [2, 1]\n"
            "assert pluck([1, 2, 3]) == [2, 1]\n"
            "assert pluck([]) == []\n"
            "assert pluck([5, 0, 3, 0, 4, 2]) == [0, 1]\n"
            "assert pluck([1, 3, 5]) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/69",
        "prompt": (
            "def search(lst):\n"
            '    """\n'
            "    You are given a non-empty list of positive integers. Return the greatest integer that is greater than\n"
            "    zero, and has a frequency greater than or equal to the value of the integer itself.\n"
            "    The frequency of an integer is the number of times it appears in the list.\n"
            "    If no such a value exist, return -1.\n"
            "    Examples:\n"
            "        search([4, 1, 2, 2, 3, 1]) == 2\n"
            "        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3\n"
            "        search([5, 5, 4, 4, 4]) == -1\n"
            '    """\n'
        ),
        "entry_point": "search",
        "test": (
            "assert search([4, 1, 2, 2, 3, 1]) == 2\n"
            "assert search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3\n"
            "assert search([5, 5, 4, 4, 4]) == -1\n"
        ),
    },
    {
        "task_id": "HumanEval/70",
        "prompt": (
            "def strange_sort_list(lst):\n"
            '    """\n'
            "    Given list of integers, return list in strange order.\n"
            "    Strange sorting, is when you start with the minimum value,\n"
            "    then maximum of the remaining integers, then minimum and so on.\n\n"
            "    Examples:\n"
            "    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]\n"
            "    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]\n"
            "    strange_sort_list([]) == []\n"
            '    """\n'
        ),
        "entry_point": "strange_sort_list",
        "test": (
            "assert strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]\n"
            "assert strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]\n"
            "assert strange_sort_list([]) == []\n"
        ),
    },
    {
        "task_id": "HumanEval/71",
        "prompt": (
            "def triangle_area(a, b, c):\n"
            '    """\n'
            "    Given the lengths of the three sides of a triangle. Return the area of\n"
            "    the triangle rounded to 2 decimal points if the three sides form a valid triangle.\n"
            "    Otherwise return -1.\n"
            "    Three sides make a valid triangle when the sum of any two sides is greater\n"
            "    than the third side.\n"
            "    Example:\n"
            "    triangle_area(3, 4, 5) == 6.00\n"
            "    triangle_area(1, 2, 10) == -1\n"
            '    """\n'
        ),
        "entry_point": "triangle_area",
        "test": (
            "assert triangle_area(3, 4, 5) == 6.00\n"
            "assert triangle_area(1, 2, 10) == -1\n"
            "assert triangle_area(5, 5, 5) == round(5**2 * (3**0.5) / 4, 2)\n"
        ),
    },
    {
        "task_id": "HumanEval/72",
        "prompt": (
            "def will_it_fly(q, w):\n"
            '    """\n'
            "    Write a function that returns True if the object q will fly, and False otherwise.\n"
            "    The object q will fly if it's balanced (it is a palindromic list) and the sum of its elements is less than or equal the maximum possible weight w.\n\n"
            "    Example:\n"
            "    will_it_fly([1, 2], 5)   ➞ False\n"
            "                             # 1+2 is less than the maximum possible weight, but it's unbalanced.\n\n"
            "    will_it_fly([3, 2, 3], 1) ➞ False\n"
            "                             # it's balanced, but 3+2+3 is more than the maximum possible weight.\n\n"
            "    will_it_fly([3, 2, 3], 9) ➞ True\n"
            "                             # 3+2+3 is less than the maximum possible weight, and it's balanced.\n\n"
            "    will_it_fly([3], 5) ➞ True\n"
            "                         # 3 is less than the maximum possible weight, and it's balanced.\n"
            '    """\n'
        ),
        "entry_point": "will_it_fly",
        "test": (
            "assert will_it_fly([1, 2], 5) == False\n"
            "assert will_it_fly([3, 2, 3], 1) == False\n"
            "assert will_it_fly([3, 2, 3], 9) == True\n"
            "assert will_it_fly([3], 5) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/73",
        "prompt": (
            "def smallest_change(arr):\n"
            '    """\n'
            "    Given an array arr of integers, find the minimum number of elements that\n"
            "    need to be changed to make the array palindromic. A palindromic array is an array that\n"
            "    is read the same backwards and forwards. In one change, you can change one element to any other element.\n\n"
            "    For example:\n"
            "    smallest_change([1,2,3,5,4,7,9,6]) == 4\n"
            "    smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1\n"
            "    smallest_change([1, 2, 3, 2, 1]) == 0\n"
            '    """\n'
        ),
        "entry_point": "smallest_change",
        "test": (
            "assert smallest_change([1, 2, 3, 5, 4, 7, 9, 6]) == 4\n"
            "assert smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1\n"
            "assert smallest_change([1, 2, 3, 2, 1]) == 0\n"
        ),
    },
    {
        "task_id": "HumanEval/74",
        "prompt": (
            "def total_match(lst1, lst2):\n"
            '    """\n'
            "    Write a function that accepts two lists of strings and returns the list that has\n"
            "    total number of chars in the all strings of the list less than or equal to the other list.\n\n"
            "    if the two lists have the same number of chars, return the first list.\n\n"
            "    Examples\n"
            "    total_match([], []) ➞ []\n"
            "    total_match(['hi', 'admin'], ['hI', 'Hi']) ➞ ['hI', 'Hi']\n"
            "    total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']) ➞ ['hi', 'admin']\n"
            "    total_match(['hi', 'admin'], ['hI', 'hi', 'hi']) ➞ ['hI', 'hi', 'hi']\n"
            "    total_match(['4'], ['1', '2', '3', '4', '5']) ➞ ['4']\n"
            '    """\n'
        ),
        "entry_point": "total_match",
        "test": (
            "assert total_match([], []) == []\n"
            "assert total_match(['hi', 'admin'], ['hI', 'Hi']) == ['hI', 'Hi']\n"
            "assert total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']) == ['hi', 'admin']\n"
        ),
    },
    {
        "task_id": "HumanEval/75",
        "prompt": (
            "def is_multiply_prime(a):\n"
            '    """Write a function that returns true if the given number is the multiplication of 3 prime numbers\n'
            "    and false otherwise.\n"
            "    Knowing that (a) is less then 100.\n"
            "    Example:\n"
            "    is_multiply_prime(30) == True\n"
            "    30 = 2 * 3 * 5\n"
            '    """\n'
        ),
        "entry_point": "is_multiply_prime",
        "test": (
            "assert is_multiply_prime(30) == True\n"
            "assert is_multiply_prime(8) == True\n"
            "assert is_multiply_prime(10) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/76",
        "prompt": (
            "def is_simple_power(x, n):\n"
            '    """Your task is to write a function that returns true if a number x is a simple\n'
            "    power of n and false in other cases.\n"
            "    x is a simple power of n if n**int=x\n"
            "    For example:\n"
            "    is_simple_power(1, 4) => true\n"
            "    is_simple_power(2, 2) => true\n"
            "    is_simple_power(8, 2) => true\n"
            "    is_simple_power(3, 2) => false\n"
            "    is_simple_power(3, 1) => false\n"
            "    is_simple_power(5, 3) => false\n"
            '    """\n'
        ),
        "entry_point": "is_simple_power",
        "test": (
            "assert is_simple_power(1, 4) == True\n"
            "assert is_simple_power(2, 2) == True\n"
            "assert is_simple_power(8, 2) == True\n"
            "assert is_simple_power(3, 2) == False\n"
            "assert is_simple_power(3, 1) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/77",
        "prompt": (
            "def iscube(a):\n"
            '    """\n'
            "    Write a function that takes an integer a and returns True \n"
            "    if this ingeger is a cube of some integer number.\n"
            "    Note: you may assume the input is always valid.\n"
            "    Examples:\n"
            "    iscube(1) ==> True\n"
            "    iscube(2) ==> False\n"
            "    iscube(-1) ==> True\n"
            "    iscube(64) ==> True\n"
            "    iscube(0) ==> True\n"
            "    iscube(180) ==> False\n"
            '    """\n'
        ),
        "entry_point": "iscube",
        "test": (
            "assert iscube(1) == True\n"
            "assert iscube(2) == False\n"
            "assert iscube(-1) == True\n"
            "assert iscube(64) == True\n"
            "assert iscube(0) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/78",
        "prompt": (
            "def hex_key(num):\n"
            '    """You have been tasked to write a function that receives\n'
            "    a hexadecimal number as a string and counts the number of hexadecimal\n"
            "    digits that are primes (prime number, or a prime, is a natural number\n"
            "    greater than 1 that is not a product of two smaller natural numbers).\n"
            "    Hexadecimal digits are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F.\n"
            "    Prime numbers are 2, 3, 5, 7, 11, 13, 17,...\n"
            "    So you have to determine a number of the following digits: 2, 3, 5, 7,\n"
            "    B (=decimal 11), D (=decimal 13).\n"
            "    Note: you may assume the input is always correct or empty string,\n"
            "    and symbols A,B,C,D,E,F are always uppercase.\n"
            "    Examples:\n"
            "    For num = '10' the output should be 1.\n"
            "    For num = 'AF' the output should be 1.\n"
            "    For num = '1077E' the output should be 2.\n"
            "    For num = 'ABED1A33' the output should be 4.\n"
            "    For num = '123456789ABCDEF0' the output should be 6.\n"
            "    For num = '2020' the output should be 2.\n"
            '    """\n'
        ),
        "entry_point": "hex_key",
        "test": (
            "assert hex_key('10') == 1\n"
            "assert hex_key('AF') == 1\n"
            "assert hex_key('1077E') == 2\n"
            "assert hex_key('ABED1A33') == 4\n"
            "assert hex_key('2020') == 2\n"
        ),
    },
    {
        "task_id": "HumanEval/79",
        "prompt": (
            "def decimal_to_binary(decimal):\n"
            '    """You will be given a number in decimal form and your task is to convert it to\n'
            "    binary format. The function should return a string, with each character representing a binary\n"
            "    number. Each character in the string will be '0' or '1'.\n\n"
            "    There will be an extra couple of characters 'db' at the beginning and at the end of the string.\n"
            "    The extra characters are there to help with the format.\n\n"
            "    Examples:\n"
            "    decimal_to_binary(15)   # returns '1111' => 'db1111db'\n"
            "    decimal_to_binary(32)   # returns '100000' => 'db100000db'\n"
            '    """\n'
        ),
        "entry_point": "decimal_to_binary",
        "test": (
            "assert decimal_to_binary(15) == 'db1111db'\n"
            "assert decimal_to_binary(32) == 'db100000db'\n"
            "assert decimal_to_binary(0) == 'db0db'\n"
        ),
    },
    {
        "task_id": "HumanEval/80",
        "prompt": (
            "def is_happy(s):\n"
            '    """You are given a string s.\n'
            "    Your task is to check if the string is happy or not.\n"
            "    A string is happy if its length is at least 3 and every 3 consecutive letters are distinct\n"
            "    Example:\n"
            "    is_happy(a) => False\n"
            "    is_happy(aa) => False\n"
            "    is_happy(abcd) => True\n"
            "    is_happy(aabb) => False\n"
            "    is_happy(adb) => True\n"
            "    is_happy(xyy) => False\n"
            '    """\n'
        ),
        "entry_point": "is_happy",
        "test": (
            "assert is_happy('a') == False\n"
            "assert is_happy('aa') == False\n"
            "assert is_happy('abcd') == True\n"
            "assert is_happy('aabb') == False\n"
            "assert is_happy('adb') == True\n"
            "assert is_happy('xyy') == False\n"
        ),
    },
    {
        "task_id": "HumanEval/81",
        "prompt": (
            "def numerical_letter_grade(grades):\n"
            '    """It is the last week of the semester and the teacher has to give the grades\n'
            "    to students. The teacher has been making her own algorithm for grading.\n"
            "    The only problem is, she has lost the code she used for grading.\n"
            "    She has given you a list of GPAs for some students and you have to write\n"
            "    a function that can output a list of letter grades using the following table:\n"
            "             GPA       |    Letter grade\n"
            "              4.0                A+\n"
            "            > 3.7               A\n"
            "            > 3.3               A-\n"
            "            > 3.0               B+\n"
            "            > 2.7               B\n"
            "            > 2.3               B-\n"
            "            > 2.0               C+\n"
            "            > 1.7               C\n"
            "            > 1.3               C-\n"
            "            > 1.0               D+\n"
            "            > 0.7               D\n"
            "            > 0.0               D-\n"
            "              0.0               E\n"
            "    Example:\n"
            "    grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']\n"
            '    """\n'
        ),
        "entry_point": "numerical_letter_grade",
        "test": (
            "assert numerical_letter_grade([4.0, 3, 1.7, 2, 3.5]) == ['A+', 'B', 'C-', 'C', 'A-']\n"
            "assert numerical_letter_grade([0.0]) == ['E']\n"
            "assert numerical_letter_grade([4.0]) == ['A+']\n"
        ),
    },
    {
        "task_id": "HumanEval/82",
        "prompt": (
            "def prime_length(string):\n"
            '    """Write a function that takes a string and returns True if the string\n'
            "    length is a prime number or False otherwise\n"
            "    Examples\n"
            "    prime_length('Hello') == True\n"
            "    prime_length('abcdcba') == True\n"
            "    prime_length('kittens') == True\n"
            "    prime_length('orange') == False\n"
            '    """\n'
        ),
        "entry_point": "prime_length",
        "test": (
            "assert prime_length('Hello') == True\n"
            "assert prime_length('abcdcba') == True\n"
            "assert prime_length('kittens') == True\n"
            "assert prime_length('orange') == False\n"
        ),
    },
    {
        "task_id": "HumanEval/83",
        "prompt": (
            "def starts_one_ends(n):\n"
            '    """\n'
            "    Given a positive integer n, return the count of the numbers of n-digit\n"
            "    positive integers that start or end with 1.\n"
            '    """\n'
        ),
        "entry_point": "starts_one_ends",
        "test": (
            "assert starts_one_ends(1) == 1\n"
            "assert starts_one_ends(2) == 18\n"
            "assert starts_one_ends(3) == 180\n"
        ),
    },
    {
        "task_id": "HumanEval/84",
        "prompt": (
            "def solve(N):\n"
            '    """Given a positive integer N, return the total sum of its digits in binary.\n\n'
            "    Example\n"
            '        For N = 1000, the sum of digits will be 1 the output should be "1".\n'
            '        For N = 150, the sum of digits will be 6 the output should be "110".\n'
            '        For N = 147, the sum of digits will be 12 the output should be "1100".\n\n'
            "    Variables:\n"
            "        @N integer\n"
            "    Constraints: 0 <= N <= 10000.\n"
            "    Output:\n"
            "        a string of binary number\n"
            '    """\n'
        ),
        "entry_point": "solve",
        "test": (
            "assert solve(1000) == '1'\nassert solve(150) == '110'\nassert solve(147) == '1100'\n"
        ),
    },
    {
        "task_id": "HumanEval/85",
        "prompt": (
            "def add(lst):\n"
            '    """Given a non-empty list of integers lst. add the even elements that are at odd indices..\n\n\n'
            "    Examples:\n"
            "        add([4, 2, 6, 7]) ==> 2\n"
            '    """\n'
        ),
        "entry_point": "add",
        "test": (
            "assert add([4, 2, 6, 7]) == 2\n"
            "assert add([2, 3, 4]) == 0\n"
            "assert add([2, 3, 4, 5, 6]) == 6\n"
        ),
    },
    {
        "task_id": "HumanEval/86",
        "prompt": (
            "def anti_shuffle(s):\n"
            '    """\n'
            "    Write a function that takes a string and returns an ordered version of it.\n"
            "    Ordered version of string, is a string where all words (separated by space)\n"
            "    are replaced by a new word where all the characters arranged in\n"
            "    ascending order based on ascii value.\n"
            "    Note: You should keep the order of words and blank spaces in the sentence.\n\n"
            "    For example:\n"
            "    anti_shuffle('Hi') returns 'Hi'\n"
            "    anti_shuffle('hello') returns 'ehllo'\n"
            "    anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'\n"
            '    """\n'
        ),
        "entry_point": "anti_shuffle",
        "test": (
            "assert anti_shuffle('Hi') == 'Hi'\n"
            "assert anti_shuffle('hello') == 'ehllo'\n"
            "assert anti_shuffle('Hello World!!!') == 'Hello !!!Wdlor'\n"
        ),
    },
    {
        "task_id": "HumanEval/87",
        "prompt": (
            "def get_row(lst, x):\n"
            '    """\n'
            "    You are given a 2 dimensional data, as a nested lists,\n"
            "    which is similar to matrix, however, unlike matrices,\n"
            "    each row may contain a different number of columns.\n"
            "    Given lst, and integer x, find integers x in the list,\n"
            "    and return list of tuples, [(x1, y1), (x2, y2) ...] such that\n"
            "    each tuple is a coordinates - (row, columns), starting with 0.\n"
            "    Sort coordinates initially by rows in ascending order.\n"
            "    Also, sort coordinates of the row by columns in descending order.\n\n"
            "    Examples:\n"
            "    get_row([\n"
            "      [1,2,3,4,5,6],\n"
            "      [1,2,3,4,1,6],\n"
            "      [1,2,3,4,5,1]\n"
            "    ], 1) == [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]\n"
            "    get_row([], 1) == []\n"
            "    get_row([[], [1], [1, 2, 3]], 3) == [(2, 2)]\n"
            '    """\n'
        ),
        "entry_point": "get_row",
        "test": (
            "assert get_row([[1,2,3,4,5,6],[1,2,3,4,1,6],[1,2,3,4,5,1]], 1) == [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]\n"
            "assert get_row([], 1) == []\n"
            "assert get_row([[], [1], [1, 2, 3]], 3) == [(2, 2)]\n"
        ),
    },
    {
        "task_id": "HumanEval/88",
        "prompt": (
            "def sort_array(array):\n"
            '    """\n'
            "    Given an array of non-negative integers, return a copy of the given array after sorting,\n"
            "    you will sort the given array in ascending order if the sum( first index value, last index value) is odd,\n"
            "    or sort it in descending order if the sum( first index value, last index value) is even.\n\n"
            "    Note:\n"
            "    * don't change the given array.\n\n"
            "    Examples:\n"
            "    * sort_array([]) => []\n"
            "    * sort_array([5]) => [5]\n"
            "    * sort_array([2, 4, 3, 0, 1, 5]) => [0, 1, 2, 3, 4, 5]\n"
            "    * sort_array([2, 4, 3, 0, 1, 5, 6]) => [6, 5, 4, 3, 2, 1, 0]\n"
            '    """\n'
        ),
        "entry_point": "sort_array",
        "test": (
            "assert sort_array([]) == []\n"
            "assert sort_array([5]) == [5]\n"
            "assert sort_array([2, 4, 3, 0, 1, 5]) == [0, 1, 2, 3, 4, 5]\n"
            "assert sort_array([2, 4, 3, 0, 1, 5, 6]) == [6, 5, 4, 3, 2, 1, 0]\n"
        ),
    },
    {
        "task_id": "HumanEval/89",
        "prompt": (
            "def encrypt(s):\n"
            '    """Create a function encrypt that takes a string as an argument and\n'
            "    returns a string encrypted with the alphabet being rotated.\n"
            "    The alphabet should be rotated in a manner such that the letters\n"
            "    shift down by two multiplied to two places.\n"
            "    For example:\n"
            "    encrypt('hi') returns 'lm'\n"
            "    encrypt('asdfghjkl') returns 'ewhjklnop'\n"
            "    encrypt('gf') returns 'kj'\n"
            "    encrypt('et') returns 'ix'\n"
            '    """\n'
        ),
        "entry_point": "encrypt",
        "test": (
            "assert encrypt('hi') == 'lm'\n"
            "assert encrypt('asdfghjkl') == 'ewhjklnop'\n"
            "assert encrypt('gf') == 'kj'\n"
            "assert encrypt('et') == 'ix'\n"
        ),
    },
    {
        "task_id": "HumanEval/90",
        "prompt": (
            "def next_smallest(lst):\n"
            '    """\n'
            "    You are given a list of integers.\n"
            "    Write a function next_smallest() that returns the 2nd smallest element of the list.\n"
            "    Return None if there is no such element.\n"
            "    \n"
            "    next_smallest([1, 2, 3, 4, 5]) == 2\n"
            "    next_smallest([5, 1, 4, 3, 2]) == 2\n"
            "    next_smallest([]) == None\n"
            "    next_smallest([1, 1]) == None\n"
            '    """\n'
        ),
        "entry_point": "next_smallest",
        "test": (
            "assert next_smallest([1, 2, 3, 4, 5]) == 2\n"
            "assert next_smallest([5, 1, 4, 3, 2]) == 2\n"
            "assert next_smallest([]) is None\n"
            "assert next_smallest([1, 1]) is None\n"
        ),
    },
    {
        "task_id": "HumanEval/91",
        "prompt": (
            "def is_bored(S):\n"
            '    """\n'
            "    You'll be given a string of words, and your task is to count the number\n"
            '    of boredoms. A boredom is a sentence that starts with the word "I".\n'
            "    Sentences are delimited by '.', '?' or '!'.\n"
            "\n"
            "    For example:\n"
            "    >>> is_bored('Hello world')\n"
            "    0\n"
            "    >>> is_bored('The sky is blue. The sun is shining. I love this weather')\n"
            "    1\n"
            '    """\n'
        ),
        "entry_point": "is_bored",
        "test": (
            "assert is_bored('Hello world') == 0\n"
            "assert is_bored('The sky is blue. The sun is shining. I love this weather') == 1\n"
            "assert is_bored('I am happy. I like cats') == 2\n"
        ),
    },
    {
        "task_id": "HumanEval/92",
        "prompt": (
            "def any_int(x, y, z):\n"
            '    """\n'
            "    Create a function that takes 3 numbers.\n"
            "    Returns true if one of the numbers is equal to the sum of the other two, and all numbers are integers.\n"
            "    Returns false in any other cases.\n"
            "\n"
            "    Examples\n"
            "    any_int(5, 2, 7) ➞ True\n"
            "    any_int(3, 2, 2) ➞ False\n"
            "    any_int(3, -2, 1) ➞ True\n"
            "    any_int(3.6, -2.2, 2) ➞ False\n"
            '    """\n'
        ),
        "entry_point": "any_int",
        "test": (
            "assert any_int(5, 2, 7) == True\n"
            "assert any_int(3, 2, 2) == False\n"
            "assert any_int(3, -2, 1) == True\n"
            "assert any_int(3.6, -2.2, 2) == False\n"
        ),
    },
    {
        "task_id": "HumanEval/93",
        "prompt": (
            "def encode(message):\n"
            '    """\n'
            "    Write a function that takes a message, and encodes in such a \n"
            "    way that it swaps case of all letters, replaces all vowels in\n"
            "    the message with the letter that appears 2 places ahead of that\n"
            "    vowel in the english alphabet.\n"
            "    Assume only letters.\n"
            "\n"
            "    Examples:\n"
            "    >>> encode('test')\n"
            "    'TGST'\n"
            "    >>> encode('This is a message')\n"
            "    'tHKS KS C MGSSCGG'\n"
            '    """\n'
        ),
        "entry_point": "encode",
        "test": (
            "assert encode('test') == 'TGST'\n"
            "assert encode('This is a message') == 'tHKS KS C MGSSCGG'\n"
        ),
    },
    {
        "task_id": "HumanEval/94",
        "prompt": (
            "def skjkasdkd(lst):\n"
            '    """You are given a list of integers.\n'
            "    You need to find the largest prime value and return the sum of its digits.\n\n"
            "    Examples:\n"
            "    For lst = [0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3] the output should be 10\n"
            "    For lst = [1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1] the output should be 25\n"
            "    For lst = [1,3,1,32,5107,34,83278,109,163,23,2323,32,30,1,9,3] the output should be 13\n"
            "    For lst = [0,724,32,71,99,32,6,0,5,91,83,0,5,6] the output should be 11\n"
            "    For lst = [0,81,12,3,1,21] the output should be 3\n"
            "    For lst = [0,8,1,2,1,7] the output should be 7\n"
            '    """\n'
        ),
        "entry_point": "skjkasdkd",
        "test": (
            "assert skjkasdkd([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3]) == 10\n"
            "assert skjkasdkd([1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1]) == 25\n"
            "assert skjkasdkd([0,8,1,2,1,7]) == 7\n"
        ),
    },
    {
        "task_id": "HumanEval/95",
        "prompt": (
            "def check_dict_case(dict):\n"
            '    """\n'
            "    Given a dictionary, return True if all keys are strings in lower\n"
            "    case or all keys are strings in upper case, else return False.\n"
            "    The function should return False is the given dictionary is empty.\n"
            "    Examples:\n"
            "    check_dict_case({'a': 'apple', 'b': 'banana'}) should return True.\n"
            "    check_dict_case({'a': 'apple', 'A': 'banana', 'B': 'banana'}) should return False.\n"
            "    check_dict_case({'a': 'apple', 8: 'banana', 'a': 'apple'}) should return False.\n"
            "    check_dict_case({'Name': 'John', 'Age': '36', 'City': 'Houston'}) should return False.\n"
            "    check_dict_case({'STATE': 'NC', 'ZIP': '12345'}) should return True.\n"
            '    """\n'
        ),
        "entry_point": "check_dict_case",
        "test": (
            "assert check_dict_case({'a': 'apple', 'b': 'banana'}) == True\n"
            "assert check_dict_case({'a': 'apple', 'A': 'banana', 'B': 'banana'}) == False\n"
            "assert check_dict_case({}) == False\n"
            "assert check_dict_case({'STATE': 'NC', 'ZIP': '12345'}) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/96",
        "prompt": (
            "def count_up_to(n):\n"
            '    """Implement a function that takes an non-negative integer and returns an array of the first n\n'
            "    integers that are prime numbers and less than n.\n"
            "    for example:\n"
            "    count_up_to(5) => [2,3]\n"
            "    count_up_to(11) => [2,3,5,7]\n"
            "    count_up_to(0) => []\n"
            "    count_up_to(20) => [2,3,5,7,11,13,17,19]\n"
            "    count_up_to(1) => []\n"
            "    count_up_to(18) => [2,3,5,7,11,13,17]\n"
            '    """\n'
        ),
        "entry_point": "count_up_to",
        "test": (
            "assert count_up_to(5) == [2, 3]\n"
            "assert count_up_to(11) == [2, 3, 5, 7]\n"
            "assert count_up_to(0) == []\n"
            "assert count_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]\n"
        ),
    },
    {
        "task_id": "HumanEval/97",
        "prompt": (
            "def multiply(a, b):\n"
            '    """Complete the function that takes two integers and returns\n'
            "    the product of their unit digits.\n"
            "    Assume the input is always valid.\n"
            "    Examples:\n"
            "    multiply(148, 412) should return 16.\n"
            "    multiply(19, 28) should return 72.\n"
            "    multiply(2020, 1851) should return 0.\n"
            "    multiply(14,-15) should return 20.\n"
            '    """\n'
        ),
        "entry_point": "multiply",
        "test": (
            "assert multiply(148, 412) == 16\n"
            "assert multiply(19, 28) == 72\n"
            "assert multiply(2020, 1851) == 0\n"
            "assert multiply(14, -15) == 20\n"
        ),
    },
    {
        "task_id": "HumanEval/98",
        "prompt": (
            "def count_upper(s):\n"
            '    """\n'
            "    Given a string s, count the number of uppercase vowels in even indices.\n"
            "\n"
            "    For example:\n"
            "    count_upper('aBCdEf') returns 1\n"
            "    count_upper('abcdefg') returns 0\n"
            "    count_upper('dBBE') returns 0\n"
            '    """\n'
        ),
        "entry_point": "count_upper",
        "test": (
            "assert count_upper('aBCdEf') == 1\n"
            "assert count_upper('abcdefg') == 0\n"
            "assert count_upper('dBBE') == 0\n"
            "assert count_upper('AEIOU') == 3\n"
        ),
    },
    {
        "task_id": "HumanEval/99",
        "prompt": (
            "def closest_integer(value):\n"
            '    """\n'
            "    Create a function that takes a value (string) representing a number\n"
            "    and returns the closest integer to it. If the number is equidistant\n"
            "    from two integers, round it away from zero.\n\n"
            "    Examples\n"
            "    >>> closest_integer('10')\n"
            "    10\n"
            "    >>> closest_integer('15.3')\n"
            "    15\n\n"
            "    Note:\n"
            "    Rounding away from zero means that if the given number is equidistant\n"
            "    from two integers, the one you should return is the one that is the\n"
            "    farthest from zero. For example closest_integer('14.5') should\n"
            "    return 15 and closest_integer('-14.5') should return -15.\n"
            '    """\n'
        ),
        "entry_point": "closest_integer",
        "test": (
            "assert closest_integer('10') == 10\n"
            "assert closest_integer('15.3') == 15\n"
            "assert closest_integer('14.5') == 15\n"
            "assert closest_integer('-14.5') == -15\n"
        ),
    },
]

assert len(_CODE_PROBLEMS_50_99) == 50, (
    f"Expected 50 extra problems, got {len(_CODE_PROBLEMS_50_99)}"
)

# ---------------------------------------------------------------------------
# 50 GSM8K-style math problems (inline subset)
# These are arithmetic reasoning problems with a known integer answer.
# ---------------------------------------------------------------------------

_MATH_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4933600. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market?",
        "answer": 18,
        "expected_str": "18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": 3,
        "expected_str": "3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": 70000,
        "expected_str": "70000",
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": 540,
        "expected_str": "540",
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "answer": 20,
        "expected_str": "20",
    },
    {
        "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "answer": 64,
        "expected_str": "64",
    },
    {
        "question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
        "answer": 340,
        "expected_str": "340",
    },
    {
        "question": "A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30 in the third month. How many total downloads occurred, if the download increase in the fourth month was half of the increase that occurred between the first and second months?",
        "answer": 510,
        "expected_str": "510",
    },
    {
        "question": "Ann's favorite store was having a summer clearance. For $75 she bought 5 pairs of shorts at $7 each and 2 pairs of shoes at $10 each. She also bought 4 tops, all at the same price. How much did each top cost?",
        "answer": 5,
        "expected_str": "5",
    },
    {
        "question": "Tom's brother is 4 times as old as Tom's dog. If in 6 years, Tom's brother will be 30 years old, how old is Tom's dog going to be in 6 years?",
        "answer": 12,
        "expected_str": "12",
    },
    {
        "question": "A farmer has 46 chickens. Each chicken gives 6 eggs per week. If the farmer sells a dozen eggs for $3, how much money will the farmer make in 8 weeks?",
        "answer": 552,
        "expected_str": "552",
    },
    {
        "question": "Mario needs to buy 40 apples for his fruit salad. He has already bought 12 green apples and 14 red apples. How many more apples does he need to buy?",
        "answer": 14,
        "expected_str": "14",
    },
    {
        "question": "There are 3 buses operating on a route. Each bus makes 4 trips per day. Each trip can carry 50 passengers. How many passengers can the route carry in 5 days?",
        "answer": 3000,
        "expected_str": "3000",
    },
    {
        "question": "A store sells pencils for $0.50 each and notebooks for $2.00 each. If a student buys 6 pencils and 3 notebooks, how much do they spend?",
        "answer": 9,
        "expected_str": "9",
    },
    {
        "question": "John runs 3 miles per day on weekdays and 5 miles per day on weekends. How many miles does he run in 2 weeks?",
        "answer": 50,
        "expected_str": "50",
    },
    {
        "question": "A classroom has 30 students. 18 are boys and the rest are girls. If 4 girls transfer out, how many girls remain?",
        "answer": 8,
        "expected_str": "8",
    },
    {
        "question": "Sam has $45. He spends $12 on lunch and $8 on a book. How much money does he have left?",
        "answer": 25,
        "expected_str": "25",
    },
    {
        "question": "A factory produces 120 toys per hour. It operates 8 hours a day. How many toys does it produce in 5 days?",
        "answer": 4800,
        "expected_str": "4800",
    },
    {
        "question": "There are 5 shelves in a bookstore. Each shelf holds 32 books. If 47 books are sold, how many books remain?",
        "answer": 113,
        "expected_str": "113",
    },
    {
        "question": "A recipe requires 3 cups of flour to make 24 cookies. How many cups of flour are needed to make 72 cookies?",
        "answer": 9,
        "expected_str": "9",
    },
    {
        "question": "A train travels at 60 km/h. How far does it travel in 2.5 hours?",
        "answer": 150,
        "expected_str": "150",
    },
    {
        "question": "Lisa earns $12 per hour. She works 7 hours a day for 5 days. How much does she earn in total?",
        "answer": 420,
        "expected_str": "420",
    },
    {
        "question": "A box contains 6 red balls and 9 blue balls. If 4 balls are removed, how many balls remain in the box?",
        "answer": 11,
        "expected_str": "11",
    },
    {
        "question": "Mike reads 25 pages per day. How many pages does he read in 3 weeks?",
        "answer": 525,
        "expected_str": "525",
    },
    {
        "question": "A garden has 8 rows of flowers with 15 flowers in each row. If 30 flowers are picked, how many remain?",
        "answer": 90,
        "expected_str": "90",
    },
    {
        "question": "There are 4 teams in a tournament. Each team plays every other team twice. How many games are played in total?",
        "answer": 12,
        "expected_str": "12",
    },
    {
        "question": "A store bought 200 items at $3 each and sold them at $5 each. What is the total profit?",
        "answer": 400,
        "expected_str": "400",
    },
    {
        "question": "Jake has 3 times as many marbles as Tom. Tom has 15 marbles. How many marbles do they have together?",
        "answer": 60,
        "expected_str": "60",
    },
    {
        "question": "A swimming pool is filled at a rate of 150 gallons per hour. How long does it take to fill 900 gallons?",
        "answer": 6,
        "expected_str": "6",
    },
    {
        "question": "A baker makes 5 cakes per hour. Each cake requires 2 cups of sugar. If the baker works for 4 hours, how many cups of sugar are needed?",
        "answer": 40,
        "expected_str": "40",
    },
    {
        "question": "There are 100 students in a school. 60% participate in sports. How many students do NOT participate in sports?",
        "answer": 40,
        "expected_str": "40",
    },
    {
        "question": "A car uses 8 liters of fuel per 100 km. How many liters does it use for a 350 km trip?",
        "answer": 28,
        "expected_str": "28",
    },
    {
        "question": "Emma has 5 packs of stickers. Each pack has 20 stickers. She gives away 45 stickers. How many does she have left?",
        "answer": 55,
        "expected_str": "55",
    },
    {
        "question": "A classroom has 6 rows of desks with 5 desks in each row. 8 new desks are added. How many desks are there now?",
        "answer": 38,
        "expected_str": "38",
    },
    {
        "question": "A pizza is cut into 8 slices. 3 friends each eat 2 slices. How many slices remain?",
        "answer": 2,
        "expected_str": "2",
    },
    {
        "question": "A library has 500 books. 120 are checked out. Of the remaining books, 60 are in reference section. How many books can be borrowed?",
        "answer": 320,
        "expected_str": "320",
    },
    {
        "question": "Tom saves $25 per week. After 8 weeks, he spends $70 on a game. How much does he have left?",
        "answer": 130,
        "expected_str": "130",
    },
    {
        "question": "A farmer plants corn in 12 rows with 30 plants per row. He also plants wheat in 8 rows with 25 plants per row. How many plants are there in total?",
        "answer": 560,
        "expected_str": "560",
    },
    {
        "question": "Each student in a class of 24 needs 3 pencils for an exam. The teacher already has 12 pencils. How many more pencils does she need to buy?",
        "answer": 60,
        "expected_str": "60",
    },
    {
        "question": "A store has 50 red shirts and 30 blue shirts. They sell 35 red and 20 blue shirts. How many shirts remain?",
        "answer": 25,
        "expected_str": "25",
    },
    {
        "question": "David cycles 4 km every morning. How many km does he cycle in the month of February in a non-leap year?",
        "answer": 112,
        "expected_str": "112",
    },
    {
        "question": "A bucket holds 12 liters. A tap fills it at 3 liters per minute. How long does it take to fill 4 buckets?",
        "answer": 16,
        "expected_str": "16",
    },
    {
        "question": "Helen has 4 boxes of chocolates. Each box has 15 chocolates. She eats 2 chocolates a day. How many days will the chocolates last?",
        "answer": 30,
        "expected_str": "30",
    },
    {
        "question": "A school trip has 4 buses, each carrying 45 students. If 12 students are absent, how many students go on the trip?",
        "answer": 168,
        "expected_str": "168",
    },
    {
        "question": "An apple costs $0.75. A banana costs $0.50. How much does it cost to buy 4 apples and 6 bananas?",
        "answer": 6,
        "expected_str": "6",
    },
    {
        "question": "A worker paints 3 walls per day. There are 36 walls to paint. How many days will it take?",
        "answer": 12,
        "expected_str": "12",
    },
    {
        "question": "A number is tripled, then 15 is subtracted. The result is 45. What is the number?",
        "answer": 20,
        "expected_str": "20",
    },
    {
        "question": "Peter earns $150 per week and spends $90 per week. After 6 weeks, how much has he saved?",
        "answer": 360,
        "expected_str": "360",
    },
    {
        "question": "A tank has 200 liters of water. 30 liters are used each day. After how many days will the tank be empty?",
        "answer": 6,
        "expected_str": "6",
    },
    {
        "question": "A classroom has 28 students. Half are boys. 3 girls are absent today. How many girls are present?",
        "answer": 11,
        "expected_str": "11",
    },
]

assert len(_MATH_PROBLEMS) == 50, f"Expected 50 math problems, got {len(_MATH_PROBLEMS)}"

# ---------------------------------------------------------------------------
# Energy scorer
# ---------------------------------------------------------------------------


class _TokenLengthEnergyScorer:
    """Fallback energy scorer when the Ising model is unavailable.

    Shorter responses are assigned lower energy — a rough proxy for the
    Ising energy heuristic.  This is NOT suitable for headline claims
    but keeps the pipeline functional when JAX is unavailable.
    """

    def score(self, text: str) -> float:
        """Return word count as a proxy energy (shorter code = lower energy = better)."""
        return float(len(text.split()))


def _build_energy_scorer() -> tuple[Any, str]:
    """Try to load the Ising energy scorer; fall back to token-length heuristic.

    The Ising scorer is JAX-based (carnot.models.ising).  If JAX is not
    installed or the model fails to initialise, we fall back to a simple
    word-count heuristic.  The fallback is clearly labelled in the artifact
    so the retrospective knows which path ran.
    """
    try:
        from carnot.models.ising import IsingConfig, IsingModel  # noqa: PLC0415
        import jax.random as jrandom  # noqa: PLC0415

        config = IsingConfig(input_dim=64, coupling_init="xavier_uniform")
        model = IsingModel(config, key=jrandom.PRNGKey(967))

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
# LLM runner wrappers — identical to Exp 906 pattern
# ---------------------------------------------------------------------------


class _LlamaCppRunner:
    """LLM runner backed by llama-cpp-python.

    Used for SOTA GGUF models.  This is the required path for any result
    that will be cited as a headline benchmark number (CLAUDE.md SOTA rule).
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        """Return the model's text completion of prompt, stripping whitespace."""
        output = self._model(prompt, max_tokens=512, temperature=0.0, echo=False)
        return output["choices"][0]["text"].strip()


class _TransformersRunner:
    """LLM runner backed by HuggingFace transformers.

    CPU-capable fallback path.  Artifacts produced with this runner are
    labelled 'fallback_model_used' so the retrospective can verify quality.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        """Generate a completion using HuggingFace transformers pipeline."""
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


def _try_load_gguf(hf_id: str) -> tuple[Any, str] | None:
    """Try to load a GGUF model via llama.cpp from the HF cache.

    Returns (runner, model_id) on success, None if the model is not cached
    or llama.cpp is unavailable.  Does not raise — failures fall through to
    the next candidate model.
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
                    print(f"[exp967] Loading GGUF {hf_id} from {model_path} …", flush=True)
                    llm = Llama(
                        model_path=str(model_path),
                        n_gpu_layers=-1,
                        n_ctx=4096,
                        verbose=False,
                    )
                    return _LlamaCppRunner(llm), hf_id
        return None
    except Exception as exc:
        print(f"[exp967] GGUF load for {hf_id} failed: {exc}", flush=True)
        return None


def _load_model() -> tuple[Any, str, bool]:
    """Load primary (Gemma-4-31B), fallback (Qwen3.6-35B), or tiny transformers model.

    Returns (runner, model_id, is_fallback).
    is_fallback=True means the tiny transformers path ran — headline quality
    is degraded and the artifact will be labelled accordingly.
    """
    # Primary: gemma-4-31B-it-GGUF
    result = _try_load_gguf("gemma-4-31B-it-GGUF")
    if result is not None:
        runner, model_id = result
        return runner, model_id, False

    # Fallback: Qwen3.6-35B-A3B-GGUF
    result = _try_load_gguf("Qwen3.6-35B-A3B-GGUF")
    if result is not None:
        runner, model_id = result
        return runner, model_id, False

    # Last resort: tiny transformers model
    print("[exp967] No GGUF available — using transformers fallback.", flush=True)
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_id = "google/gemma-4-E4B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return _TransformersRunner(model, tokenizer), model_id, True


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_generation_prompt(problem: dict[str, Any]) -> str:
    """Build the initial LLM generation prompt for a HumanEval problem.

    Consistent with Exp 905/906 format for fair cross-experiment comparison.
    """
    return (
        "Complete the following Python function. "
        "Output ONLY the function body code (no imports, no prose):\n\n" + problem["prompt"]
    )


def _build_hypothesis_prompt(problem: dict[str, Any], initial_code: str, error: str) -> str:
    """Build a prompt asking the model to diagnose WHY the initial code is wrong.

    This implements the DebugRepair hypothesis step from arXiv 2604.19305.
    The model generates a natural-language hypothesis about the root cause
    BEFORE being asked to generate a fix.  This forces the model to reason
    about the failure mode explicitly, reducing the chance it regenerates
    the same bug.

    Returns the hypothesis prompt (the DIAGNOSIS step, not the fix step).
    """
    return (
        "You tried to implement this Python function:\n\n"
        f"{problem['prompt']}\n\n"
        "Your implementation was:\n"
        f"```python\n{initial_code}\n```\n\n"
        "It failed with this error:\n"
        f"```\n{error}\n```\n\n"
        "Before fixing the code, first explain in 1-2 sentences: "
        "WHY is this code wrong? What is the root cause of the failure? "
        "Output ONLY your diagnosis (no code)."
    )


def _build_debug_repair_prompt(
    problem: dict[str, Any], initial_code: str, error: str, hypothesis: str
) -> str:
    """Build a repair prompt that includes the model's own diagnosis hypothesis.

    The hypothesis (generated by _build_hypothesis_prompt) gives the model
    explicit context about what went wrong.  arXiv 2604.19305 calls this the
    'DebugRepair' step and reports +8.2pp on HumanEval over vanilla repair.

    The causal chain: name-the-bug -> fix-the-bug.  Naming it first forces
    an explicit intermediate reasoning step that prevents the model from
    regenerating the exact same logic with minor surface variations.
    """
    return (
        "Complete the following Python function. "
        "Output ONLY the function body code (no imports, no prose):\n\n"
        f"{problem['prompt']}\n\n"
        "Your previous attempt produced:\n"
        f"```python\n{initial_code}\n```\n\n"
        "It failed with:\n"
        f"```\n{error}\n```\n\n"
        f"My diagnosis of the error: {hypothesis}\n\n"
        "Now generate a corrected implementation. Return ONLY the Python code."
    )


def _build_standard_repair_prompt(problem: dict[str, Any], initial_code: str, error: str) -> str:
    """Build a standard (no-hypothesis) repair prompt for ablation comparison.

    This is the baseline repair method used in Exp 905/906.
    We run a subset of problems with this prompt and with the hypothesis
    prompt to measure hypothesis_contribution.
    """
    return (
        "Complete the following Python function. "
        "Output ONLY the function body code (no imports, no prose):\n\n"
        f"{problem['prompt']}\n\n"
        "Your previous attempt produced:\n"
        f"```python\n{initial_code}\n```\n\n"
        "It failed with:\n"
        f"```\n{error}\n```\n\n"
        "Fix the code so that it passes all the tests. "
        "Return ONLY the corrected Python code."
    )


# ---------------------------------------------------------------------------
# Code execution helper
# ---------------------------------------------------------------------------


def _exec_code(
    code: str, test_cases: list[str], timeout_s: float = 10.0
) -> tuple[bool, str | None]:
    """Execute code + test cases in a subprocess.

    Returns (passed, error_text).  error_text is None on success or a
    string containing the truncated traceback on failure.

    We use a subprocess to sandbox the execution and enforce the timeout.
    The CARNOT_USE_SANDBOX env var is not required here — the subprocess
    boundary provides basic isolation for the HumanEval test suite.
    """
    import subprocess  # noqa: PLC0415

    script = f"{code}\n\n{chr(10).join(test_cases)}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return True, None
        error = (result.stderr or result.stdout or "non-zero exit")[:2000]
        return False, error
    except subprocess.TimeoutExpired:
        return False, f"TimeoutError: execution exceeded {timeout_s}s"
    except Exception as exc:
        return False, str(exc)[:2000]


# ---------------------------------------------------------------------------
# Code domain: IterativeSelfRepair with DebugRepair
# ---------------------------------------------------------------------------


def _run_code_with_debug_repair(
    runner: Any,
    problems: list[dict[str, Any]],
    energy_scorer: Any,
    use_hypothesis: bool,
    tmpl: ExperimentTemplate,
    prefix: str,
) -> list[dict[str, Any]]:
    """Run IterativeSelfRepair with or without the DebugRepair hypothesis step.

    Parameters
    ----------
    use_hypothesis : bool
        When True, generates a diagnosis hypothesis before each repair prompt
        (arXiv 2604.19305 DebugRepair).  When False, uses the standard
        repair prompt from Exp 905/906 (ablation baseline).

    Returns list of per-problem result dicts.
    """
    results: list[dict[str, Any]] = []

    for idx, prob in enumerate(problems):
        task_id = prob["task_id"]
        gen_prompt = _build_generation_prompt(prob)
        test_cases = [line for line in prob["test"].strip().splitlines() if line.strip()]

        print(f"[exp967/{prefix}] {idx + 1}/{len(problems)}: {task_id} …", flush=True)
        t0 = time.perf_counter()

        try:
            # Attempt 0: initial generation
            initial_code = runner.generate(gen_prompt)
            baseline_passed, error = _exec_code(initial_code, test_cases)

            best_code = initial_code
            best_score = energy_scorer.score(initial_code)
            best_passed = baseline_passed
            n_retries = 0

            if not baseline_passed and error is not None:
                for retry_i in range(3):
                    n_retries += 1

                    if use_hypothesis:
                        # DebugRepair: generate hypothesis first, then repair
                        hyp_prompt = _build_hypothesis_prompt(prob, best_code, error)
                        hypothesis = runner.generate(hyp_prompt)
                        repair_prompt = _build_debug_repair_prompt(
                            prob, best_code, error, hypothesis
                        )
                    else:
                        repair_prompt = _build_standard_repair_prompt(prob, best_code, error)

                    repaired_code = runner.generate(repair_prompt)
                    passed, new_error = _exec_code(repaired_code, test_cases)
                    score = energy_scorer.score(repaired_code)

                    if score < best_score:
                        best_code = repaired_code
                        best_score = score
                        best_passed = passed

                    if passed:
                        break
                    if new_error is not None:
                        error = new_error

            elapsed = round(time.perf_counter() - t0, 2)
            print(
                f"[exp967/{prefix}]   baseline={baseline_passed} repair={best_passed} "
                f"retries={n_retries} energy={best_score:.3f} [{elapsed}s]",
                flush=True,
            )

            results.append(
                {
                    "task_id": task_id,
                    "baseline_passed": baseline_passed,
                    "repair_passed": best_passed,
                    "n_retries": n_retries,
                    "energy_score_best": best_score,
                    "elapsed_s": elapsed,
                    "use_hypothesis": use_hypothesis,
                }
            )

        except Exception as exc:
            print(f"[exp967/{prefix}]   ERROR: {exc}", flush=True)
            results.append(
                {
                    "task_id": task_id,
                    "error": str(exc),
                    "baseline_passed": False,
                    "repair_passed": False,
                    "n_retries": 0,
                    "energy_score_best": 0.0,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "use_hypothesis": use_hypothesis,
                }
            )

        if (idx + 1) % 10 == 0:
            tmpl.checkpoint_save({f"{prefix}_results": results}, step=idx + 1)

    return results


# ---------------------------------------------------------------------------
# Math domain: GSM8K-style problems with external feedback loop
# ---------------------------------------------------------------------------


def _check_math_answer(response: str, expected_str: str) -> bool:
    """Check if the response contains the expected numeric answer.

    Looks for the expected number as a standalone token anywhere in the
    response.  This is a lenient check — the model may output the answer
    in various formats (e.g., "The answer is 42" or just "42").
    """
    import re  # noqa: PLC0415

    clean = response.replace(",", "").strip()
    pattern = rf"\b{re.escape(expected_str)}\b"
    return bool(re.search(pattern, clean))


def _run_math_repair(
    runner: Any,
    problems: list[dict[str, Any]],
    tmpl: ExperimentTemplate,
) -> list[dict[str, Any]]:
    """Run math repair loop: initial attempt + up to 3 feedback-based retries.

    No energy scoring for the math domain — we just check if the numeric
    answer appears in the response.  When wrong, we tell the model it got
    the wrong answer and ask it to try again (external feedback re-feeding).

    This matches the description in the task spec: "no energy scoring here,
    just external feedback re-feeding".
    """
    results: list[dict[str, Any]] = []

    for idx, prob in enumerate(problems):
        question = prob["question"]
        expected = prob["expected_str"]

        t0 = time.perf_counter()
        print(f"[exp967/math] {idx + 1}/{len(problems)}: GSM8K Q{idx + 1} …", flush=True)

        try:
            prompt = (
                "Solve the following math problem step by step. "
                "End your answer with 'The answer is X.' where X is the numeric answer.\n\n"
                + question
            )

            response = runner.generate(prompt)
            baseline_passed = _check_math_answer(response, expected)

            best_passed = baseline_passed
            n_retries = 0

            if not baseline_passed:
                for _ in range(3):
                    n_retries += 1
                    # Feed back that the answer was wrong and ask for retry
                    retry_prompt = (
                        f"{question}\n\n"
                        f"Your previous answer was incorrect. "
                        f"Please reconsider and solve again step by step. "
                        f"End with 'The answer is X.'\n\n"
                        f"Previous attempt: {response[:200]}"
                    )
                    response = runner.generate(retry_prompt)
                    if _check_math_answer(response, expected):
                        best_passed = True
                        break

            elapsed = round(time.perf_counter() - t0, 2)
            print(
                f"[exp967/math]   baseline={baseline_passed} repair={best_passed} "
                f"retries={n_retries} [{elapsed}s]",
                flush=True,
            )

            results.append(
                {
                    "question_idx": idx,
                    "baseline_passed": baseline_passed,
                    "repair_passed": best_passed,
                    "n_retries": n_retries,
                    "elapsed_s": elapsed,
                }
            )

        except Exception as exc:
            print(f"[exp967/math]   ERROR: {exc}", flush=True)
            results.append(
                {
                    "question_idx": idx,
                    "error": str(exc),
                    "baseline_passed": False,
                    "repair_passed": False,
                    "n_retries": 0,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
            )

        if (idx + 1) % 10 == 0:
            tmpl.checkpoint_save({"math_results": results}, step=idx + 1)

    return results


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _pass_rates(results: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return (baseline_pass_rate, repair_pass_rate, delta) for a result list."""
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
    n_base = sum(1 for r in results if r.get("baseline_passed", False))
    n_rep = sum(1 for r in results if r.get("repair_passed", False))
    base_rate = n_base / n
    rep_rate = n_rep / n
    return base_rate, rep_rate, rep_rate - base_rate


def _compute_hypothesis_contribution(
    without_hyp_results: list[dict[str, Any]],
    with_hyp_results: list[dict[str, Any]],
) -> float:
    """Estimate hypothesis_contribution as pass_rate(with_hyp) - pass_rate(without_hyp).

    We run the first 20 code problems both ways (ablation subset) to isolate
    the marginal contribution of the DebugRepair hypothesis step.
    pass_rate is measured on the REPAIR column (after repair), not baseline.

    A positive value means the hypothesis step helps; negative means it hurts.
    The arXiv 2604.19305 paper reports +8.2pp on HumanEval with GPT-4.
    """
    n = min(len(without_hyp_results), len(with_hyp_results))
    if n == 0:
        return 0.0
    without_rate = sum(1 for r in without_hyp_results[:n] if r.get("repair_passed", False)) / n
    with_rate = sum(1 for r in with_hyp_results[:n] if r.get("repair_passed", False)) / n
    return round(with_rate - without_rate, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate Exp 967: IterativeSelfRepair 100q + DebugRepair hypothesis step."""
    tmpl = ExperimentTemplate(
        exp_id=967,
        title="IterativeSelfRepair 100q + DebugRepair Hypothesis Step",
        deliverable=_DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    t_start = time.perf_counter()

    # ---- Energy scorer -------------------------------------------------------
    energy_scorer, energy_scorer_type = _build_energy_scorer()
    print(f"[exp967] Energy scorer: {energy_scorer_type}", flush=True)

    # ---- Load 100 HumanEval problems -----------------------------------------
    try:
        from human_eval.data import read_problems  # noqa: PLC0415

        all_problems = read_problems()
        task_ids = sorted(all_problems.keys())[:100]
        code_problems: list[dict[str, Any]] = [all_problems[tid] for tid in task_ids]
        print(f"[exp967] Loaded {len(code_problems)} problems from human_eval package.", flush=True)
    except ImportError:
        # Build the 100-problem inline set by merging Exp 906's 50 + our 50
        import importlib.util  # noqa: PLC0415

        spec906_path = _REPO_ROOT / "scripts" / "experiment_906_code_repair_50q_scaleup.py"
        spec906 = importlib.util.spec_from_file_location("exp906", spec906_path)
        assert spec906 is not None
        mod906 = importlib.util.module_from_spec(spec906)
        spec906.loader.exec_module(mod906)  # type: ignore[union-attr]
        code_problems = mod906._INLINE_PROBLEMS + _CODE_PROBLEMS_50_99
        print(
            f"[exp967] human_eval not installed — using inline problems ({len(code_problems)}).",
            flush=True,
        )

    n_code = len(code_problems)

    # ---- Load primary model --------------------------------------------------
    try:
        runner, model_id, is_fallback = _load_model()
    except Exception as exc:
        print(f"[exp967] Model load failed: {exc}", flush=True)
        artifact = tmpl.build_result(
            {"model_load_error": str(exc), "traceback": tb.format_exc()},
            status="blocked",
            honest_verdict="blocked_model_load_failure",
        )
        artifact.update(
            {
                "code_repair_delta": 0.0,
                "math_repair_delta": 0.0,
                "hypothesis_contribution": 0.0,
                "n_problems_code": 0,
                "n_problems_math": 0,
                "model_used": "none",
            }
        )
        (_REPO_ROOT / _DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    print(f"[exp967] Using model: {model_id} (fallback={is_fallback})", flush=True)

    # ---- CPU fallback: cap problem count so the experiment finishes in ~30 min --
    # The GGUF path (llama.cpp) runs ~5s/problem.  The transformers fallback on CPU
    # runs ~900s/problem.  When on fallback we cap at 3 ablation + 7 main code + 5
    # math problems to stay within a 90-minute budget.  The artifact is labelled
    # 'cpu_fallback_limited' so the retrospective knows headline quality is degraded.
    if is_fallback:
        n_code_limit = 10
        n_math_limit = 5
        code_problems = code_problems[:n_code_limit]
        n_code = n_code_limit
        print(
            f"[exp967] CPU fallback detected — capping at {n_code_limit} code + "
            f"{n_math_limit} math problems to meet time budget.",
            flush=True,
        )
    else:
        n_math_limit = len(_MATH_PROBLEMS)

    # ---- Ablation subset: run first N problems with AND without hypothesis --
    # This gives us hypothesis_contribution without doubling the runtime.
    ablation_n = min(3 if is_fallback else 20, n_code)
    ablation_problems = code_problems[:ablation_n]

    print(f"[exp967] Running ablation ({ablation_n} problems, without hypothesis) …", flush=True)
    without_hyp_results = _run_code_with_debug_repair(
        runner,
        ablation_problems,
        energy_scorer,
        use_hypothesis=False,
        tmpl=tmpl,
        prefix="ablation_no_hyp",
    )

    print(f"[exp967] Running ablation ({ablation_n} problems, with hypothesis) …", flush=True)
    with_hyp_results = _run_code_with_debug_repair(
        runner,
        ablation_problems,
        energy_scorer,
        use_hypothesis=True,
        tmpl=tmpl,
        prefix="ablation_with_hyp",
    )

    hypothesis_contribution = _compute_hypothesis_contribution(
        without_hyp_results, with_hyp_results
    )
    print(
        f"[exp967] Hypothesis contribution (ablation): {hypothesis_contribution:+.4f}", flush=True
    )

    # ---- Full 100q run with hypothesis (DebugRepair) -------------------------
    # Skip the first 20 (already covered by ablation with_hyp_results).
    remaining_problems = code_problems[ablation_n:]
    print(
        f"[exp967] Running remaining {len(remaining_problems)} problems with DebugRepair …",
        flush=True,
    )
    remaining_results = _run_code_with_debug_repair(
        runner,
        remaining_problems,
        energy_scorer,
        use_hypothesis=True,
        tmpl=tmpl,
        prefix="main_with_hyp",
    )

    # Full 100q code results = ablation with_hyp + remaining
    all_code_results = with_hyp_results + remaining_results
    code_base, code_repair, code_delta = _pass_rates(all_code_results)

    print(
        f"[exp967] CODE: baseline={code_base:.3f} repair={code_repair:.3f} delta={code_delta:+.3f}",
        flush=True,
    )

    # ---- Math domain: 50 GSM8K problems --------------------------------------
    math_problems_to_run = _MATH_PROBLEMS[:n_math_limit]
    print(f"[exp967] Running {len(math_problems_to_run)} math problems …", flush=True)
    math_results = _run_math_repair(runner, math_problems_to_run, tmpl)
    math_base, math_repair, math_delta = _pass_rates(math_results)

    print(
        f"[exp967] MATH: baseline={math_base:.3f} repair={math_repair:.3f} delta={math_delta:+.3f}",
        flush=True,
    )

    # ---- Honest verdict ------------------------------------------------------
    if code_delta > 0 or math_delta > 0:
        honest_verdict = "iterative_repair_100q_viable"
    else:
        honest_verdict = "no_improvement_100q"

    duration_s = round(time.perf_counter() - t_start, 2)

    print(
        f"\n[exp967] verdict={honest_verdict}  hyp_contribution={hypothesis_contribution:+.4f}  "
        f"duration={duration_s}s",
        flush=True,
    )

    # ---- Write artifact (required schema fields + extras) --------------------
    inference_mode = "fallback_transformers_limited" if is_fallback else "live_gguf"

    artifact = tmpl.build_result(
        {
            # Required schema fields
            "code_repair_delta": round(code_delta, 4),
            "math_repair_delta": round(math_delta, 4),
            "hypothesis_contribution": hypothesis_contribution,
            "n_problems_code": n_code,
            "n_problems_math": len(_MATH_PROBLEMS),
            "model_used": model_id,
            "honest_verdict": honest_verdict,
            # Additional fields for retrospective analysis
            "code_baseline_pass_rate": round(code_base, 4),
            "code_repair_pass_rate": round(code_repair, 4),
            "math_baseline_pass_rate": round(math_base, 4),
            "math_repair_pass_rate": round(math_repair, 4),
            "energy_scorer_type": energy_scorer_type,
            "ablation_n": ablation_n,
            "inference_mode": inference_mode,
            "max_retries": 3,
            "exec_timeout_s": 10.0,
            "code_results_per_problem": all_code_results,
            "math_results_per_problem": math_results,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode=inference_mode,
    )

    # Ensure all required schema fields appear at top level for easy parsing
    for key in (
        "code_repair_delta",
        "math_repair_delta",
        "hypothesis_contribution",
        "n_problems_code",
        "n_problems_math",
        "model_used",
        "honest_verdict",
    ):
        if key not in artifact:
            artifact[key] = artifact.get(key, 0.0 if "delta" in key or "contribution" in key else 0)

    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[exp967] Artifact written to {output_path}", flush=True)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
