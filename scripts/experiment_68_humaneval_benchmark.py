#!/usr/bin/env python3
"""Experiment 68: HumanEval-style code verification benchmark.

**Researcher summary:**
    Evaluates the full Carnot code verification pipeline on 50 HumanEval-style
    coding problems. Combines constraint extraction (Exp 48), runtime
    instrumentation (Exp 53), and Ising-guided fuzzing (Exp 54) into a unified
    code verification pipeline, then measures pass@1 and pass@1+repair rates.

**Detailed explanation for engineers:**
    HumanEval is OpenAI's standard benchmark for code generation. Each problem
    provides a function signature, docstring, and test cases. This experiment
    runs 50 HumanEval-style problems through the following pipeline:

    1. **Generate**: Produce a code solution (via Qwen3.5-0.8B or simulation).
    2. **Extract constraints**: Parse the solution's AST to find type
       annotations, bounds, return types, and initialization issues (Exp 48).
    3. **Instrument**: Insert runtime isinstance/bound/return checks (Exp 53).
    4. **Test**: Run the provided canonical test cases against the solution.
    5. **Fuzz**: Run 20 Ising-guided fuzz inputs (Exp 54) to find bugs that
       the canonical tests miss — boundary values, empty inputs, negatives.
    6. **Repair**: If any test or fuzz failure, feed failures back to the LLM
       for up to 3 repair iterations (Exp 57 pattern).

    Metrics:
    - pass@1: fraction passing all tests on first generation
    - pass@1+repair: fraction passing after verify-repair loop (max 3 iters)
    - Bug detection breakdown: test-only, instrumentation-only, fuzzing-only
    - Unique bugs found by fuzzing that canonical tests missed

    If the LLM or HumanEval dataset cannot be loaded, the experiment falls back
    to 50 manually-crafted problems with known-buggy simulated solutions. The
    pipeline logic is fully exercised either way.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_68_humaneval_benchmark.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import ast
import gc
import math
import os
import random
import re
import sys
import textwrap
import time
import traceback
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_48_code_constraints import code_to_constraints
from experiment_53_runtime_constraints import instrument_code, execute_instrumented


# ---------------------------------------------------------------------------
# 1. HumanEval-style problem definitions
# ---------------------------------------------------------------------------

def load_humaneval_problems() -> list[dict[str, Any]]:
    """Load HumanEval problems. Falls back to manual definitions if unavailable.

    **Detailed explanation for engineers:**
        First tries ``from human_eval.data import read_problems`` (the official
        HumanEval package). If that import fails (common — it requires a pip
        install from the OpenAI repo), we generate 50 manually-crafted problems
        that mirror HumanEval's format:

        Each problem has:
        - ``task_id``: unique identifier (e.g., "HumanEval/0")
        - ``prompt``: function signature + docstring (what the LLM sees)
        - ``canonical_solution``: the correct implementation
        - ``test_cases``: list of (input_args, expected_output) tuples
        - ``entry_point``: the function name to call

        The 50 manual problems cover the same categories as HumanEval:
        string manipulation, math, list operations, and simple algorithms.
    """
    try:
        from human_eval.data import read_problems
        problems_dict = read_problems()
        problems = []
        for task_id, p in list(problems_dict.items())[:50]:
            # Parse test cases from the test string if available.
            test_cases = _parse_humaneval_tests(p.get("test", ""), p["entry_point"])
            problems.append({
                "task_id": task_id,
                "prompt": p["prompt"],
                "canonical_solution": p["canonical_solution"],
                "test_cases": test_cases,
                "entry_point": p["entry_point"],
            })
        print(f"  Loaded {len(problems)} problems from human_eval package.")
        return problems
    except (ImportError, Exception) as e:
        print(f"  human_eval not available ({e}), using manual problems.")
        return _create_manual_problems()


def _parse_humaneval_tests(test_str: str, entry_point: str) -> list[tuple[list, Any]]:
    """Best-effort parse of HumanEval test strings into (args, expected) tuples.

    **Detailed explanation for engineers:**
        HumanEval test strings are Python code with assert statements like:
            assert candidate(1, 2) == 3
        We parse these to extract the call arguments and expected result.
        Falls back to empty list if parsing fails — the problem will still
        be tested via instrumentation and fuzzing.
    """
    cases: list[tuple[list, Any]] = []
    try:
        for line in test_str.strip().split("\n"):
            line = line.strip()
            if not line.startswith("assert"):
                continue
            # Try to extract: assert func(args) == expected
            match = re.match(
                r"assert\s+candidate\((.+?)\)\s*==\s*(.+?)(?:\s*$|\s*,)",
                line,
            )
            if match:
                args_str = match.group(1)
                expected_str = match.group(2).strip()
                try:
                    args = eval(f"[{args_str}]")  # noqa: S307
                    expected = eval(expected_str)  # noqa: S307
                    cases.append((args, expected))
                except Exception:
                    pass
    except Exception:
        pass
    return cases


def _create_manual_problems() -> list[dict[str, Any]]:
    """Create 50 HumanEval-style problems manually.

    **Detailed explanation for engineers:**
        Each problem has a function signature, docstring with examples,
        a canonical (correct) solution, and test cases. Problems span four
        categories to match HumanEval's distribution:

        - String manipulation (12 problems): reverse, palindrome, vowel count,
          anagram detection, character frequency, etc.
        - Math/number theory (13 problems): factorial, fibonacci, GCD, prime
          check, digit sum, power of two, etc.
        - List/array operations (13 problems): max element, flatten, rotate,
          remove duplicates, two-sum, merge sorted, etc.
        - Simple algorithms (12 problems): binary search, bubble sort, balanced
          parens, run-length encoding, Caesar cipher, etc.

        The test cases include normal inputs AND edge cases (empty strings,
        zero, negative numbers, single-element lists) to ensure meaningful
        coverage even before Ising fuzzing adds more.
    """
    problems: list[dict[str, Any]] = []
    idx = 0

    def _add(name: str, prompt: str, solution: str,
             tests: list[tuple[list, Any]], entry: str) -> None:
        nonlocal idx
        problems.append({
            "task_id": f"HumanEval/{idx}",
            "prompt": textwrap.dedent(prompt),
            "canonical_solution": textwrap.dedent(solution),
            "test_cases": tests,
            "entry_point": entry,
        })
        idx += 1

    # ---- String manipulation (12) ----

    _add("reverse_string", """
        def reverse_string(s: str) -> str:
            \"\"\"Return the reverse of string s.
            >>> reverse_string("hello")
            'olleh'
            >>> reverse_string("")
            ''
            \"\"\"
        """, """
            return s[::-1]
        """, [
        (["hello"], "olleh"), ([""], ""), (["a"], "a"),
        (["abcde"], "edcba"), (["racecar"], "racecar"),
    ], "reverse_string")

    _add("is_palindrome", """
        def is_palindrome(s: str) -> bool:
            \"\"\"Check if s is a palindrome (case-insensitive).
            >>> is_palindrome("racecar")
            True
            >>> is_palindrome("hello")
            False
            \"\"\"
        """, """
            s = s.lower()
            return s == s[::-1]
        """, [
        (["racecar"], True), (["hello"], False), ([""], True),
        (["Aba"], True), (["ab"], False),
    ], "is_palindrome")

    _add("count_vowels", """
        def count_vowels(s: str) -> int:
            \"\"\"Count the number of vowels (a, e, i, o, u) in s.
            >>> count_vowels("hello")
            2
            >>> count_vowels("xyz")
            0
            \"\"\"
        """, """
            return sum(1 for c in s.lower() if c in 'aeiou')
        """, [
        (["hello"], 2), (["xyz"], 0), ([""], 0),
        (["aeiou"], 5), (["HELLO"], 2),
    ], "count_vowels")

    _add("capitalize_words", """
        def capitalize_words(s: str) -> str:
            \"\"\"Capitalize the first letter of each word in s.
            >>> capitalize_words("hello world")
            'Hello World'
            >>> capitalize_words("")
            ''
            \"\"\"
        """, """
            return s.title()
        """, [
        (["hello world"], "Hello World"), ([""], ""),
        (["a"], "A"), (["hello"], "Hello"),
    ], "capitalize_words")

    _add("count_words", """
        def count_words(s: str) -> int:
            \"\"\"Count the number of words in s (split by whitespace).
            >>> count_words("hello world")
            2
            >>> count_words("")
            0
            \"\"\"
        """, """
            return len(s.split()) if s.strip() else 0
        """, [
        (["hello world"], 2), ([""], 0), (["  "], 0),
        (["one"], 1), (["a b c d"], 4),
    ], "count_words")

    _add("remove_duplicates_str", """
        def remove_duplicates_str(s: str) -> str:
            \"\"\"Remove duplicate characters from s, keeping first occurrence.
            >>> remove_duplicates_str("aabbcc")
            'abc'
            >>> remove_duplicates_str("")
            ''
            \"\"\"
        """, """
            seen = set()
            result = []
            for c in s:
                if c not in seen:
                    seen.add(c)
                    result.append(c)
            return ''.join(result)
        """, [
        (["aabbcc"], "abc"), ([""], ""), (["abc"], "abc"),
        (["aaaa"], "a"), (["abba"], "ab"),
    ], "remove_duplicates_str")

    _add("char_frequency", """
        def char_frequency(s: str) -> dict:
            \"\"\"Return a dict mapping each character to its frequency.
            >>> char_frequency("aab")
            {'a': 2, 'b': 1}
            \"\"\"
        """, """
            freq = {}
            for c in s:
                freq[c] = freq.get(c, 0) + 1
            return freq
        """, [
        (["aab"], {"a": 2, "b": 1}), ([""], {}),
        (["abc"], {"a": 1, "b": 1, "c": 1}),
    ], "char_frequency")

    _add("longest_word", """
        def longest_word(s: str) -> str:
            \"\"\"Return the longest word in s. If tie, return the first.
            >>> longest_word("the quick brown fox")
            'quick'
            >>> longest_word("")
            ''
            \"\"\"
        """, """
            words = s.split()
            if not words:
                return ''
            return max(words, key=len)
        """, [
        (["the quick brown fox"], "quick"), ([""], ""),
        (["a"], "a"), (["hi there"], "there"),
    ], "longest_word")

    _add("is_anagram", """
        def is_anagram(s1: str, s2: str) -> bool:
            \"\"\"Check if s1 and s2 are anagrams (case-insensitive).
            >>> is_anagram("listen", "silent")
            True
            >>> is_anagram("hello", "world")
            False
            \"\"\"
        """, """
            return sorted(s1.lower()) == sorted(s2.lower())
        """, [
        (["listen", "silent"], True), (["hello", "world"], False),
        (["", ""], True), (["a", "a"], True), (["ab", "ba"], True),
    ], "is_anagram")

    _add("caesar_cipher", """
        def caesar_cipher(s: str, shift: int) -> str:
            \"\"\"Encrypt s using Caesar cipher with given shift (a-z only).
            >>> caesar_cipher("abc", 1)
            'bcd'
            >>> caesar_cipher("xyz", 3)
            'abc'
            \"\"\"
        """, """
            result = []
            for c in s:
                if c.isalpha():
                    base = ord('A') if c.isupper() else ord('a')
                    result.append(chr((ord(c) - base + shift) % 26 + base))
                else:
                    result.append(c)
            return ''.join(result)
        """, [
        (["abc", 1], "bcd"), (["xyz", 3], "abc"),
        (["", 5], ""), (["ABC", 1], "BCD"),
    ], "caesar_cipher")

    _add("run_length_encode", """
        def run_length_encode(s: str) -> str:
            \"\"\"Run-length encode a string.
            >>> run_length_encode("aaabbc")
            'a3b2c1'
            >>> run_length_encode("")
            ''
            \"\"\"
        """, """
            if not s:
                return ''
            result = []
            count = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1]:
                    count += 1
                else:
                    result.append(f'{s[i-1]}{count}')
                    count = 1
            result.append(f'{s[-1]}{count}')
            return ''.join(result)
        """, [
        (["aaabbc"], "a3b2c1"), ([""], ""), (["a"], "a1"),
        (["aaa"], "a3"),
    ], "run_length_encode")

    _add("compress_string", """
        def compress_string(s: str) -> str:
            \"\"\"Compress string by removing consecutive duplicate characters.
            >>> compress_string("aaabbc")
            'abc'
            >>> compress_string("")
            ''
            \"\"\"
        """, """
            if not s:
                return ''
            result = [s[0]]
            for c in s[1:]:
                if c != result[-1]:
                    result.append(c)
            return ''.join(result)
        """, [
        (["aaabbc"], "abc"), ([""], ""), (["abc"], "abc"),
        (["aaa"], "a"),
    ], "compress_string")

    # ---- Math/number theory (13) ----

    _add("factorial", """
        def factorial(n: int) -> int:
            \"\"\"Compute n! (n factorial). n >= 0.
            >>> factorial(5)
            120
            >>> factorial(0)
            1
            \"\"\"
        """, """
            if n <= 1:
                return 1
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result
        """, [
        ([5], 120), ([0], 1), ([1], 1), ([3], 6), ([10], 3628800),
    ], "factorial")

    _add("fibonacci", """
        def fibonacci(n: int) -> int:
            \"\"\"Return the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.
            >>> fibonacci(6)
            8
            >>> fibonacci(0)
            0
            \"\"\"
        """, """
            if n <= 0:
                return 0
            if n == 1:
                return 1
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        """, [
        ([6], 8), ([0], 0), ([1], 1), ([2], 1), ([10], 55),
    ], "fibonacci")

    _add("gcd", """
        def gcd(a: int, b: int) -> int:
            \"\"\"Compute the greatest common divisor of a and b.
            >>> gcd(12, 8)
            4
            >>> gcd(7, 13)
            1
            \"\"\"
        """, """
            while b:
                a, b = b, a % b
            return abs(a)
        """, [
        ([12, 8], 4), ([7, 13], 1), ([0, 5], 5),
        ([100, 75], 25), ([1, 1], 1),
    ], "gcd")

    _add("is_prime", """
        def is_prime(n: int) -> bool:
            \"\"\"Check if n is a prime number.
            >>> is_prime(7)
            True
            >>> is_prime(4)
            False
            \"\"\"
        """, """
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        """, [
        ([7], True), ([4], False), ([1], False), ([2], True),
        ([0], False), ([13], True), ([15], False),
    ], "is_prime")

    _add("digit_sum", """
        def digit_sum(n: int) -> int:
            \"\"\"Return the sum of digits of n (absolute value).
            >>> digit_sum(123)
            6
            >>> digit_sum(-45)
            9
            \"\"\"
        """, """
            n = abs(n)
            total = 0
            while n > 0:
                total += n % 10
                n //= 10
            return total
        """, [
        ([123], 6), ([-45], 9), ([0], 0), ([9], 9), ([999], 27),
    ], "digit_sum")

    _add("power_of_two", """
        def power_of_two(n: int) -> bool:
            \"\"\"Check if n is a power of two.
            >>> power_of_two(8)
            True
            >>> power_of_two(6)
            False
            \"\"\"
        """, """
            if n <= 0:
                return False
            return (n & (n - 1)) == 0
        """, [
        ([8], True), ([6], False), ([1], True), ([0], False),
        ([16], True), ([-4], False),
    ], "power_of_two")

    _add("lcm", """
        def lcm(a: int, b: int) -> int:
            \"\"\"Compute the least common multiple of a and b.
            >>> lcm(4, 6)
            12
            >>> lcm(3, 7)
            21
            \"\"\"
        """, """
            if a == 0 or b == 0:
                return 0
            def _gcd(x, y):
                while y:
                    x, y = y, x % y
                return abs(x)
            return abs(a * b) // _gcd(a, b)
        """, [
        ([4, 6], 12), ([3, 7], 21), ([0, 5], 0), ([1, 1], 1),
    ], "lcm")

    _add("is_perfect_square", """
        def is_perfect_square(n: int) -> bool:
            \"\"\"Check if n is a perfect square.
            >>> is_perfect_square(16)
            True
            >>> is_perfect_square(15)
            False
            \"\"\"
        """, """
            if n < 0:
                return False
            root = int(n ** 0.5)
            return root * root == n
        """, [
        ([16], True), ([15], False), ([0], True), ([1], True),
        ([-1], False), ([25], True),
    ], "is_perfect_square")

    _add("decimal_to_binary", """
        def decimal_to_binary(n: int) -> str:
            \"\"\"Convert non-negative integer n to binary string.
            >>> decimal_to_binary(10)
            '1010'
            >>> decimal_to_binary(0)
            '0'
            \"\"\"
        """, """
            if n == 0:
                return '0'
            bits = []
            while n > 0:
                bits.append(str(n % 2))
                n //= 2
            return ''.join(reversed(bits))
        """, [
        ([10], "1010"), ([0], "0"), ([1], "1"), ([255], "11111111"),
    ], "decimal_to_binary")

    _add("abs_value", """
        def abs_value(n: int) -> int:
            \"\"\"Return the absolute value of n without using abs().
            >>> abs_value(-5)
            5
            >>> abs_value(3)
            3
            \"\"\"
        """, """
            return n if n >= 0 else -n
        """, [
        ([-5], 5), ([3], 3), ([0], 0), ([-100], 100),
    ], "abs_value")

    _add("clamp", """
        def clamp(n: int, lo: int, hi: int) -> int:
            \"\"\"Clamp n between lo and hi.
            >>> clamp(5, 0, 10)
            5
            >>> clamp(-5, 0, 10)
            0
            \"\"\"
        """, """
            return max(lo, min(n, hi))
        """, [
        ([5, 0, 10], 5), ([-5, 0, 10], 0), ([15, 0, 10], 10),
        ([0, 0, 0], 0),
    ], "clamp")

    _add("sum_of_squares", """
        def sum_of_squares(n: int) -> int:
            \"\"\"Return 1^2 + 2^2 + ... + n^2. n >= 0.
            >>> sum_of_squares(3)
            14
            >>> sum_of_squares(0)
            0
            \"\"\"
        """, """
            return sum(i * i for i in range(1, n + 1))
        """, [
        ([3], 14), ([0], 0), ([1], 1), ([5], 55),
    ], "sum_of_squares")

    _add("is_leap_year", """
        def is_leap_year(year: int) -> bool:
            \"\"\"Check if a year is a leap year.
            >>> is_leap_year(2000)
            True
            >>> is_leap_year(1900)
            False
            \"\"\"
        """, """
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        """, [
        ([2000], True), ([1900], False), ([2024], True),
        ([2023], False), ([400], True),
    ], "is_leap_year")

    # ---- List operations (13) ----

    _add("find_max", """
        def find_max(arr: list) -> int:
            \"\"\"Return the maximum element in arr. arr is non-empty.
            >>> find_max([3, 1, 4, 1, 5])
            5
            \"\"\"
        """, """
            return max(arr)
        """, [
        ([[3, 1, 4, 1, 5]], 5), ([[1]], 1), ([[-1, -2, -3]], -1),
        ([[0, 0, 0]], 0),
    ], "find_max")

    _add("list_sum", """
        def list_sum(arr: list) -> int:
            \"\"\"Return the sum of all elements in arr.
            >>> list_sum([1, 2, 3])
            6
            >>> list_sum([])
            0
            \"\"\"
        """, """
            return sum(arr)
        """, [
        ([[1, 2, 3]], 6), ([[]], 0), ([[5]], 5), ([[-1, 1]], 0),
    ], "list_sum")

    _add("flatten_list", """
        def flatten_list(lst: list) -> list:
            \"\"\"Flatten a nested list one level deep.
            >>> flatten_list([[1, 2], [3, 4]])
            [1, 2, 3, 4]
            >>> flatten_list([])
            []
            \"\"\"
        """, """
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        """, [
        ([[[1, 2], [3, 4]]], [1, 2, 3, 4]), ([[]], []),
        ([[[1], [], [2, 3]]], [1, 2, 3]),
    ], "flatten_list")

    _add("remove_duplicates", """
        def remove_duplicates(arr: list) -> list:
            \"\"\"Remove duplicates from arr, preserving order.
            >>> remove_duplicates([1, 2, 2, 3, 1])
            [1, 2, 3]
            >>> remove_duplicates([])
            []
            \"\"\"
        """, """
            seen = set()
            result = []
            for x in arr:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
            return result
        """, [
        ([[1, 2, 2, 3, 1]], [1, 2, 3]), ([[]], []),
        ([[1, 1, 1]], [1]), ([[1, 2, 3]], [1, 2, 3]),
    ], "remove_duplicates")

    _add("rotate_list", """
        def rotate_list(arr: list, k: int) -> list:
            \"\"\"Rotate arr to the right by k positions.
            >>> rotate_list([1, 2, 3, 4, 5], 2)
            [4, 5, 1, 2, 3]
            >>> rotate_list([], 3)
            []
            \"\"\"
        """, """
            if not arr:
                return []
            k = k % len(arr)
            return arr[-k:] + arr[:-k] if k else list(arr)
        """, [
        ([[1, 2, 3, 4, 5], 2], [4, 5, 1, 2, 3]), ([[], 3], []),
        ([[1], 5], [1]), ([[1, 2], 0], [1, 2]),
    ], "rotate_list")

    _add("two_sum", """
        def two_sum(nums: list, target: int) -> list:
            \"\"\"Return indices of two numbers that add up to target.
            >>> two_sum([2, 7, 11, 15], 9)
            [0, 1]
            \"\"\"
        """, """
            seen = {}
            for i, n in enumerate(nums):
                complement = target - n
                if complement in seen:
                    return [seen[complement], i]
                seen[n] = i
            return []
        """, [
        ([[2, 7, 11, 15], 9], [0, 1]),
        ([[3, 3], 6], [0, 1]),
    ], "two_sum")

    _add("merge_sorted", """
        def merge_sorted(a: list, b: list) -> list:
            \"\"\"Merge two sorted lists into one sorted list.
            >>> merge_sorted([1, 3, 5], [2, 4, 6])
            [1, 2, 3, 4, 5, 6]
            >>> merge_sorted([], [1])
            [1]
            \"\"\"
        """, """
            result = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i] <= b[j]:
                    result.append(a[i])
                    i += 1
                else:
                    result.append(b[j])
                    j += 1
            result.extend(a[i:])
            result.extend(b[j:])
            return result
        """, [
        ([[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6]),
        ([[], [1]], [1]), ([[], []], []),
        ([[1], [2]], [1, 2]),
    ], "merge_sorted")

    _add("intersection", """
        def intersection(a: list, b: list) -> list:
            \"\"\"Return the intersection of two lists (unique elements, sorted).
            >>> intersection([1, 2, 3], [2, 3, 4])
            [2, 3]
            >>> intersection([], [1])
            []
            \"\"\"
        """, """
            return sorted(set(a) & set(b))
        """, [
        ([[1, 2, 3], [2, 3, 4]], [2, 3]), ([[], [1]], []),
        ([[1, 2], [3, 4]], []),
    ], "intersection")

    _add("chunk_list", """
        def chunk_list(arr: list, n: int) -> list:
            \"\"\"Split arr into chunks of size n.
            >>> chunk_list([1, 2, 3, 4, 5], 2)
            [[1, 2], [3, 4], [5]]
            >>> chunk_list([], 3)
            []
            \"\"\"
        """, """
            return [arr[i:i+n] for i in range(0, len(arr), n)] if arr else []
        """, [
        ([[1, 2, 3, 4, 5], 2], [[1, 2], [3, 4], [5]]),
        ([[], 3], []), ([[1], 1], [[1]]),
    ], "chunk_list")

    _add("filter_even", """
        def filter_even(arr: list) -> list:
            \"\"\"Return only even numbers from arr.
            >>> filter_even([1, 2, 3, 4, 5])
            [2, 4]
            >>> filter_even([])
            []
            \"\"\"
        """, """
            return [x for x in arr if x % 2 == 0]
        """, [
        ([[1, 2, 3, 4, 5]], [2, 4]), ([[]], []),
        ([[1, 3, 5]], []), ([[2, 4]], [2, 4]),
    ], "filter_even")

    _add("dot_product", """
        def dot_product(a: list, b: list) -> int:
            \"\"\"Compute the dot product of two equal-length lists.
            >>> dot_product([1, 2, 3], [4, 5, 6])
            32
            >>> dot_product([], [])
            0
            \"\"\"
        """, """
            return sum(x * y for x, y in zip(a, b))
        """, [
        ([[1, 2, 3], [4, 5, 6]], 32), ([[], []], 0),
        ([[1], [1]], 1),
    ], "dot_product")

    _add("is_sorted", """
        def is_sorted(arr: list) -> bool:
            \"\"\"Check if arr is sorted in non-decreasing order.
            >>> is_sorted([1, 2, 3])
            True
            >>> is_sorted([3, 1, 2])
            False
            \"\"\"
        """, """
            return all(arr[i] <= arr[i+1] for i in range(len(arr) - 1))
        """, [
        ([[1, 2, 3]], True), ([[3, 1, 2]], False), ([[]], True),
        ([[1]], True), ([[2, 1]], False),
    ], "is_sorted")

    _add("count_occurrences", """
        def count_occurrences(arr: list, target: int) -> int:
            \"\"\"Count how many times target appears in arr.
            >>> count_occurrences([1, 2, 2, 3], 2)
            2
            >>> count_occurrences([], 5)
            0
            \"\"\"
        """, """
            return arr.count(target)
        """, [
        ([[1, 2, 2, 3], 2], 2), ([[], 5], 0),
        ([[1, 1, 1], 1], 3), ([[1, 2, 3], 4], 0),
    ], "count_occurrences")

    # ---- Simple algorithms (12) ----

    _add("binary_search", """
        def binary_search(arr: list, target: int) -> int:
            \"\"\"Return index of target in sorted arr, or -1 if not found.
            >>> binary_search([1, 3, 5, 7, 9], 5)
            2
            >>> binary_search([1, 3, 5], 4)
            -1
            \"\"\"
        """, """
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
        """, [
        ([[1, 3, 5, 7, 9], 5], 2), ([[1, 3, 5], 4], -1),
        ([[], 1], -1), ([[5], 5], 0),
    ], "binary_search")

    _add("bubble_sort", """
        def bubble_sort(arr: list) -> list:
            \"\"\"Sort arr using bubble sort and return a new sorted list.
            >>> bubble_sort([3, 1, 4, 1, 5])
            [1, 1, 3, 4, 5]
            >>> bubble_sort([])
            []
            \"\"\"
        """, """
            arr = list(arr)
            n = len(arr)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr
        """, [
        ([[3, 1, 4, 1, 5]], [1, 1, 3, 4, 5]), ([[]], []),
        ([[1]], [1]), ([[2, 1]], [1, 2]),
    ], "bubble_sort")

    _add("balanced_parens", """
        def balanced_parens(s: str) -> bool:
            \"\"\"Check if parentheses in s are balanced.
            >>> balanced_parens("(())")
            True
            >>> balanced_parens("(()")
            False
            \"\"\"
        """, """
            count = 0
            for c in s:
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                if count < 0:
                    return False
            return count == 0
        """, [
        (["(())"], True), (["(()"], False), ([""], True),
        ([")"], False), (["()()"], True),
    ], "balanced_parens")

    _add("fizzbuzz", """
        def fizzbuzz(n: int) -> list:
            \"\"\"Return FizzBuzz list from 1 to n.
            >>> fizzbuzz(5)
            ['1', '2', 'Fizz', '4', 'Buzz']
            \"\"\"
        """, """
            result = []
            for i in range(1, n + 1):
                if i % 15 == 0:
                    result.append('FizzBuzz')
                elif i % 3 == 0:
                    result.append('Fizz')
                elif i % 5 == 0:
                    result.append('Buzz')
                else:
                    result.append(str(i))
            return result
        """, [
        ([5], ["1", "2", "Fizz", "4", "Buzz"]),
        ([0], []),
        ([15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
                "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]),
    ], "fizzbuzz")

    _add("hamming_distance", """
        def hamming_distance(s1: str, s2: str) -> int:
            \"\"\"Compute Hamming distance between equal-length strings.
            >>> hamming_distance("karolin", "kathrin")
            3
            \"\"\"
        """, """
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        """, [
        (["karolin", "kathrin"], 3), (["", ""], 0),
        (["abc", "abc"], 0), (["abc", "xyz"], 3),
    ], "hamming_distance")

    _add("first_duplicate", """
        def first_duplicate(arr: list) -> int:
            \"\"\"Return first element that appears twice, or -1 if none.
            >>> first_duplicate([1, 2, 3, 2, 1])
            2
            >>> first_duplicate([1, 2, 3])
            -1
            \"\"\"
        """, """
            seen = set()
            for x in arr:
                if x in seen:
                    return x
                seen.add(x)
            return -1
        """, [
        ([[1, 2, 3, 2, 1]], 2), ([[1, 2, 3]], -1),
        ([[]], -1), ([[1, 1]], 1),
    ], "first_duplicate")

    _add("missing_number", """
        def missing_number(arr: list, n: int) -> int:
            \"\"\"Find the missing number in arr containing 0..n with one missing.
            >>> missing_number([0, 1, 3], 3)
            2
            \"\"\"
        """, """
            expected = n * (n + 1) // 2
            return expected - sum(arr)
        """, [
        ([[0, 1, 3], 3], 2), ([[1, 2], 2], 0),
        ([[0], 1], 1),
    ], "missing_number")

    _add("majority_element", """
        def majority_element(arr: list) -> int:
            \"\"\"Return the element that appears more than n/2 times, or -1.
            >>> majority_element([1, 1, 1, 2, 3])
            1
            >>> majority_element([1, 2, 3])
            -1
            \"\"\"
        """, """
            counts = {}
            for x in arr:
                counts[x] = counts.get(x, 0) + 1
            n = len(arr)
            for val, cnt in counts.items():
                if cnt > n // 2:
                    return val
            return -1
        """, [
        ([[1, 1, 1, 2, 3]], 1), ([[1, 2, 3]], -1),
        ([[5, 5, 5]], 5),
    ], "majority_element")

    _add("max_subarray_sum", """
        def max_subarray_sum(arr: list) -> int:
            \"\"\"Find the maximum sum of a contiguous subarray (Kadane's).
            >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
            6
            >>> max_subarray_sum([])
            0
            \"\"\"
        """, """
            if not arr:
                return 0
            max_sum = current = arr[0]
            for x in arr[1:]:
                current = max(x, current + x)
                max_sum = max(max_sum, current)
            return max_sum
        """, [
        ([[-2, 1, -3, 4, -1, 2, 1, -5, 4]], 6), ([[]], 0),
        ([[1]], 1), ([[-1, -2]], -1),
    ], "max_subarray_sum")

    _add("pascal_row", """
        def pascal_row(n: int) -> list:
            \"\"\"Return the nth row of Pascal's triangle (0-indexed).
            >>> pascal_row(4)
            [1, 4, 6, 4, 1]
            >>> pascal_row(0)
            [1]
            \"\"\"
        """, """
            row = [1]
            for k in range(1, n + 1):
                row.append(row[-1] * (n - k + 1) // k)
            return row
        """, [
        ([4], [1, 4, 6, 4, 1]), ([0], [1]), ([1], [1, 1]),
    ], "pascal_row")

    _add("matrix_transpose", """
        def matrix_transpose(matrix: list) -> list:
            \"\"\"Transpose a matrix (list of lists).
            >>> matrix_transpose([[1, 2], [3, 4]])
            [[1, 3], [2, 4]]
            >>> matrix_transpose([])
            []
            \"\"\"
        """, """
            if not matrix:
                return []
            return [list(row) for row in zip(*matrix)]
        """, [
        ([[[1, 2], [3, 4]]], [[1, 3], [2, 4]]), ([[]], []),
        ([[[1]]], [[1]]),
    ], "matrix_transpose")

    _add("longest_common_prefix", """
        def longest_common_prefix(strs: list) -> str:
            \"\"\"Find the longest common prefix among a list of strings.
            >>> longest_common_prefix(["flower", "flow", "flight"])
            'fl'
            >>> longest_common_prefix([])
            ''
            \"\"\"
        """, """
            if not strs:
                return ''
            prefix = strs[0]
            for s in strs[1:]:
                while not s.startswith(prefix):
                    prefix = prefix[:-1]
                    if not prefix:
                        return ''
            return prefix
        """, [
        ([["flower", "flow", "flight"]], "fl"), ([[]], ""),
        ([["abc"]], "abc"), ([["a", "b"]], ""),
    ], "longest_common_prefix")

    assert len(problems) == 50, f"Expected 50 problems, got {len(problems)}"
    return problems


# ---------------------------------------------------------------------------
# 2. Known-buggy code solutions (for simulation when LLM is unavailable)
# ---------------------------------------------------------------------------

# Maps of entry_point -> buggy code that has a common LLM-generation mistake.
# Each buggy solution has exactly ONE intentional bug.

_BUGGY_SOLUTIONS: dict[str, tuple[str, str]] = {
    # (buggy_code, bug_description)
    "reverse_string": (
        "def reverse_string(s: str) -> str:\n    return s[1::-1]",
        "Off-by-one: only reverses first 2 chars",
    ),
    "is_palindrome": (
        "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
        "Missing case-insensitive: 'Aba' -> False instead of True",
    ),
    "count_vowels": (
        "def count_vowels(s: str) -> int:\n    return sum(1 for c in s if c in 'aeiou')",
        "Missing .lower(): uppercase vowels not counted",
    ),
    "factorial": (
        "def factorial(n: int) -> int:\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result",
        "Off-by-one: range(1, n) instead of range(1, n+1)",
    ),
    "fibonacci": (
        "def fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n):\n        a, b = b, a + b\n    return b",
        "Off-by-one: range(2, n) instead of range(2, n+1)",
    ),
    "gcd": (
        "def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a",
        "Missing abs(): negative inputs return negative GCD",
    ),
    "is_prime": (
        "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True",
        "Inefficient but correct — actually no bug here, used as control",
    ),
    "binary_search": (
        "def binary_search(arr: list, target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "Off-by-one: lo < hi instead of lo <= hi, misses single-element case",
    ),
    "bubble_sort": (
        "def bubble_sort(arr: list) -> list:\n    arr = list(arr)\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
        "Uses n-1 instead of n-i-1 in inner loop (still sorts but O(n^2) always)",
    ),
    "balanced_parens": (
        "def balanced_parens(s: str) -> bool:\n    count = 0\n    for c in s:\n        if c == '(':\n            count += 1\n        elif c == ')':\n            count -= 1\n    return count == 0",
        "Missing early return when count < 0: ')(' passes incorrectly",
    ),
    "digit_sum": (
        "def digit_sum(n: int) -> int:\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total",
        "Missing abs(): negative input infinite loops or returns 0",
    ),
    "remove_duplicates": (
        "def remove_duplicates(arr: list) -> list:\n    return list(set(arr))",
        "Doesn't preserve order: set() loses insertion ordering on some inputs",
    ),
    "rotate_list": (
        "def rotate_list(arr: list, k: int) -> list:\n    if not arr:\n        return []\n    return arr[-k:] + arr[:-k]",
        "Missing k % len(arr): k > len(arr) returns wrong result",
    ),
    "flatten_list": (
        "def flatten_list(lst: list) -> list:\n    result = []\n    for item in lst:\n        result.extend(item)\n    return result",
        "Assumes all items are lists: fails on non-list elements",
    ),
    "fizzbuzz": (
        "def fizzbuzz(n: int) -> list:\n    result = []\n    for i in range(1, n + 1):\n        if i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result",
        "Missing FizzBuzz: checks %3 and %5 separately, never produces 'FizzBuzz'",
    ),
}


# ---------------------------------------------------------------------------
# 3. LLM loading and generation
# ---------------------------------------------------------------------------

def load_llm() -> tuple[Any, Any, str, bool]:
    """Attempt to load Qwen3.5-0.8B; return (tokenizer, model, device, success).

    **Detailed explanation for engineers:**
        Tries Qwen3.5-0.8B first, then falls back to Qwen3-0.6B. Uses CUDA if
        available, otherwise CPU. Runs a subprocess smoke test with 60-second
        timeout to catch ROCm hangs on unsupported GPUs (like Radeon 890M).

        Set CARNOT_SKIP_LLM=1 to skip model loading entirely and use simulated
        outputs (useful for CI or machines without enough RAM).
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print("  CARNOT_SKIP_LLM set — skipping model loading.")
        return None, None, "cpu", False

    try:
        import subprocess
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"]:
            try:
                print(f"  Loading {model_name} on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True,
                    dtype=torch.float16 if device == "cuda" else None,
                )
                if device == "cuda":
                    model = model.cuda()
                model.eval()
                print(f"  Loaded {model_name}. Running smoke test...")

                smoke_script = (
                    f"import torch; "
                    f"from transformers import AutoModelForCausalLM, AutoTokenizer; "
                    f"t = AutoTokenizer.from_pretrained('{model_name}', trust_remote_code=True); "
                    f"m = AutoModelForCausalLM.from_pretrained('{model_name}', trust_remote_code=True); "
                    f"{'m = m.cuda(); ' if device == 'cuda' else ''}"
                    f"m.eval(); "
                    f"i = t('Hi', return_tensors='pt'); "
                    f"{'i = {{k: v.cuda() for k, v in i.items()}}; ' if device == 'cuda' else ''}"
                    f"o = m.generate(**i, max_new_tokens=4, do_sample=False, pad_token_id=t.eos_token_id); "
                    f"print('OK')"
                )
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", smoke_script],
                        timeout=60,
                        capture_output=True,
                        text=True,
                    )
                    if "OK" in result.stdout:
                        print(f"  Smoke test passed. Model ready.")
                        return tokenizer, model, device, True
                    else:
                        print(f"  Smoke test failed: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"  Smoke test timed out (60s).")
                except Exception as e:
                    print(f"  Smoke test error: {e}")

                print(f"  Falling back to simulated mode.")
                del model, tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")

    except ImportError as e:
        print(f"  torch/transformers not available: {e}")

    return None, None, "cpu", False


def generate_with_llm(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a code solution from the loaded LLM.

    **Detailed explanation for engineers:**
        Uses HuggingFace transformers generate() with greedy decoding for
        reproducibility. Applies the model's chat template and strips any
        <think>...</think> reasoning tokens from Qwen models.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


def simulate_solution(
    problem: dict[str, Any],
    iteration: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Generate a simulated LLM solution for fallback mode.

    **Detailed explanation for engineers:**
        When the real LLM can't be loaded, we simulate code generation with
        a realistic error rate:

        - ~70% of first attempts use the canonical (correct) solution.
        - ~30% use a known-buggy variant from _BUGGY_SOLUTIONS.
        - On repair iterations, 60% of buggy solutions get fixed per iteration.

        If no known-buggy variant exists for a problem, the canonical solution
        is always used (simulating a correct first attempt).
    """
    if rng is None:
        rng = random.Random(68)

    entry = problem["entry_point"]
    canonical = problem["canonical_solution"]

    # Build the full correct function.
    prompt_lines = problem["prompt"].strip().split("\n")
    # Find the function def line to get the signature.
    func_def_line = ""
    for line in prompt_lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            func_def_line = stripped
            break

    # The canonical_solution is already written with proper indentation
    # relative to the function body. We need to ensure it's indented inside
    # the function. Detect the indentation level from the prompt's last line
    # (the docstring closing) and apply it to the solution.
    prompt_stripped = problem["prompt"].strip()
    # Find the indentation of the docstring body to match solution indent.
    indent = "    "  # default 4-space indent inside function body
    for line in prompt_stripped.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            indent = line[:len(line) - len(stripped)]
            break
        if stripped.startswith(">>>"):
            indent = line[:len(line) - len(stripped)]
            break

    # Indent each line of the canonical solution to sit inside the function.
    sol_lines = canonical.strip().split("\n")
    indented_sol = "\n".join(indent + line if line.strip() else line for line in sol_lines)
    correct_code = prompt_stripped + "\n" + indented_sol

    # Decide whether to introduce a bug.
    base_error_rate = 0.30
    effective_error = base_error_rate * (0.4 ** iteration)
    should_be_buggy = rng.random() < effective_error

    if should_be_buggy and entry in _BUGGY_SOLUTIONS:
        buggy_code, _desc = _BUGGY_SOLUTIONS[entry]
        return buggy_code

    return correct_code


def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    **Detailed explanation for engineers:**
        LLMs often wrap code in markdown fences (```python ... ```) or include
        explanatory text before/after the code. This function extracts just
        the Python code by:
        1. Looking for ```python ... ``` blocks first.
        2. Falling back to ``` ... ``` blocks.
        3. If no fences found, looking for lines starting with 'def '.
        4. As a last resort, returning the entire response.
    """
    # Try to extract from markdown code fences.
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for a function definition.
    lines = response.split("\n")
    code_lines: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)
    return response


# ---------------------------------------------------------------------------
# 4. Verification pipeline: tests + instrumentation + fuzzing
# ---------------------------------------------------------------------------

def run_test_cases(
    code: str,
    entry_point: str,
    test_cases: list[tuple[list, Any]],
) -> dict[str, Any]:
    """Execute canonical test cases against the generated code.

    **Detailed explanation for engineers:**
        Compiles and executes the code in a restricted namespace, then calls
        the entry_point function with each test case's arguments and compares
        against the expected output. Catches all exceptions to prevent crashes.

    Returns:
        Dict with keys:
        - ``passed``: number of tests that passed
        - ``failed``: number that failed
        - ``errors``: list of error descriptions
        - ``all_pass``: bool, True if all tests passed
    """
    safe_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in {"open", "exec", "eval", "__import__", "compile",
                      "breakpoint", "exit", "quit"}
    } if isinstance(__builtins__, type(sys)) else {
        k: v for k, v in __builtins__.items()
        if k not in {"open", "exec", "eval", "__import__", "compile",
                      "breakpoint", "exit", "quit"}
    }

    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        return {
            "passed": 0, "failed": len(test_cases),
            "errors": [f"Code execution failed: {e}"],
            "all_pass": False,
        }

    if entry_point not in namespace:
        return {
            "passed": 0, "failed": len(test_cases),
            "errors": [f"Function '{entry_point}' not defined in code"],
            "all_pass": False,
        }

    func = namespace[entry_point]
    passed = 0
    failed = 0
    errors: list[str] = []

    for args, expected in test_cases:
        try:
            result = func(*args)
            if result == expected:
                passed += 1
            else:
                failed += 1
                errors.append(
                    f"test({args}) = {repr(result)}, expected {repr(expected)}"
                )
        except Exception as e:
            failed += 1
            errors.append(f"test({args}) raised {type(e).__name__}: {e}")

    return {
        "passed": passed, "failed": failed,
        "errors": errors, "all_pass": failed == 0,
    }


def run_instrumentation(code: str, entry_point: str) -> dict[str, Any]:
    """Instrument code and execute to check runtime constraints (Exp 53).

    **Detailed explanation for engineers:**
        Uses Exp 53's instrument_code() to insert isinstance guards, return-type
        checks, and loop-bound assertions into the AST. Then executes the
        instrumented code with a small set of probe inputs to trigger any
        runtime constraint violations.

        Returns a summary of which runtime checks were violated.
    """
    # Extract constraints statically (Exp 48).
    constraints = code_to_constraints(code)
    n_static_violations = sum(
        1 for c in constraints if c.get("satisfied") is False
    )

    # Instrument and execute with minimal probe inputs.
    instrumented = instrument_code(code)

    # Try to determine parameter types from the AST for probe generation.
    probe_inputs = _generate_probe_inputs(code, entry_point)
    if probe_inputs:
        exec_result = execute_instrumented(
            instrumented, probe_inputs, entry_point,
        )
        n_dynamic_violations = exec_result["n_fail"]
        violations = exec_result.get("violations", [])
    else:
        n_dynamic_violations = 0
        violations = []

    return {
        "n_static_violations": n_static_violations,
        "n_dynamic_violations": n_dynamic_violations,
        "static_constraints": constraints,
        "dynamic_violations": violations,
        "detected": n_static_violations > 0 or n_dynamic_violations > 0,
    }


def _generate_probe_inputs(
    code: str, entry_point: str
) -> list[dict[str, Any]]:
    """Generate a small set of probe inputs for runtime instrumentation.

    **Detailed explanation for engineers:**
        Parses the function signature to determine parameter types and generates
        a few representative inputs: zero values, small positives, empty
        collections. These probe inputs are designed to trigger runtime checks
        (isinstance guards, bound checks) without needing to know the function's
        semantics.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            params = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                ann = None
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        ann = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        ann = str(arg.annotation.value)
                params.append((arg.arg, ann))

            if not params:
                return [{}]

            # Generate a single probe input with type-appropriate defaults.
            probe: dict[str, Any] = {}
            for name, ann in params:
                if ann == "int":
                    probe[name] = 1
                elif ann == "float":
                    probe[name] = 1.0
                elif ann == "str":
                    probe[name] = "test"
                elif ann == "list":
                    probe[name] = [1, 2, 3]
                elif ann == "bool":
                    probe[name] = True
                elif ann == "dict":
                    probe[name] = {"a": 1}
                else:
                    probe[name] = 1  # fallback to int
            return [probe]

    return []


def run_ising_fuzz(
    code: str,
    entry_point: str,
    canonical_solution: str,
    prompt: str,
    n_fuzz: int = 20,
) -> dict[str, Any]:
    """Run Ising-guided fuzzing (Exp 54) to find bugs via differential testing.

    **Detailed explanation for engineers:**
        Compiles both the generated (potentially buggy) code and the canonical
        solution, then uses Ising-guided fuzzing to generate edge-case inputs
        and runs differential testing: any input where the generated code
        disagrees with the canonical solution is a detected bug.

        The Ising sampler biases toward boundary values (0, -1, empty lists,
        max values) where LLM-generated code commonly fails.

    Args:
        code: The generated (potentially buggy) code.
        entry_point: Function name to call.
        canonical_solution: The known-correct full code (prompt + solution).
        prompt: The problem prompt (to build the canonical function).
        n_fuzz: Number of fuzz inputs to generate (default 20).

    Returns:
        Dict with bugs_found, unique_sigs, fuzz_inputs tested, and errors.
    """
    from experiment_54_ising_fuzzing import (
        encode_input_space,
        decode_spins,
        differential_fuzz,
    )
    from carnot.samplers.parallel_ising import (
        ParallelIsingSampler,
        AnnealingSchedule,
    )
    import jax.numpy as jnp
    import jax.random as jrandom

    # Compile both functions in isolated namespaces.
    safe_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in {"open", "exec", "eval", "__import__", "compile",
                      "breakpoint", "exit", "quit"}
    } if isinstance(__builtins__, type(sys)) else {
        k: v for k, v in __builtins__.items()
        if k not in {"open", "exec", "eval", "__import__", "compile",
                      "breakpoint", "exit", "quit"}
    }

    ns_buggy: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    ns_ref: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}

    # Build canonical code from prompt + canonical solution.
    # Indent solution lines to sit inside the function body.
    prompt_stripped = prompt.strip()
    indent = "    "
    for line in prompt_stripped.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            indent = line[:len(line) - len(stripped)]
            break
        if stripped.startswith(">>>"):
            indent = line[:len(line) - len(stripped)]
            break
    sol_lines = canonical_solution.strip().split("\n")
    indented_sol = "\n".join(
        indent + line if line.strip() else line for line in sol_lines
    )
    canonical_code = prompt_stripped + "\n" + indented_sol

    try:
        exec(code, ns_buggy)  # noqa: S102
        exec(canonical_code, ns_ref)  # noqa: S102
    except Exception as e:
        return {
            "bugs_found": 0, "unique_sigs": set(), "fuzz_inputs": 0,
            "errors": [f"Compilation failed: {e}"],
        }

    if entry_point not in ns_buggy or entry_point not in ns_ref:
        return {
            "bugs_found": 0, "unique_sigs": set(), "fuzz_inputs": 0,
            "errors": [f"Function '{entry_point}' not defined"],
        }

    buggy_func = ns_buggy[entry_point]
    ref_func = ns_ref[entry_point]

    # Determine parameter types from the AST for Ising encoding.
    params = _get_param_types(code, entry_point)
    if not params:
        # Can't determine types — fall back to random int inputs.
        params = [("x", "int")]

    # Only fuzz parameters that are int or list_int (Ising encoding limitation).
    fuzzable_params = [
        (name, kind) for name, kind in params
        if kind in ("int", "list_int")
    ]

    if not fuzzable_params:
        # No fuzzable parameters — generate random inputs manually.
        rng = np.random.default_rng(68)
        test_inputs: list[dict[str, Any]] = []
        for _ in range(n_fuzz):
            inp: dict[str, Any] = {}
            for name, kind in params:
                if kind == "str":
                    inp[name] = "".join(
                        chr(rng.integers(97, 123)) for _ in range(rng.integers(0, 10))
                    )
                elif kind == "list":
                    length = int(rng.integers(0, 7))
                    inp[name] = [int(rng.integers(-128, 128)) for _ in range(length)]
                elif kind == "int":
                    inp[name] = int(rng.integers(-128, 128))
                else:
                    inp[name] = 1
            test_inputs.append(inp)

        result = differential_fuzz(buggy_func, ref_func, test_inputs)
        return {
            "bugs_found": result.bugs_found,
            "unique_sigs": result.unique_sigs,
            "fuzz_inputs": result.total_inputs,
            "errors": [(e[0], e[1], e[2]) for e in result.errors[:5]],
        }

    # Ising-guided fuzzing for int/list_int parameters.
    try:
        biases, J, encodings = encode_input_space(fuzzable_params)
        sampler = ParallelIsingSampler(
            n_warmup=100,
            n_samples=n_fuzz,
            steps_per_sample=5,
            schedule=AnnealingSchedule(beta_init=0.5, beta_final=5.0),
            use_checkerboard=True,
        )
        key = jrandom.PRNGKey(68)
        samples = sampler.sample(
            key,
            jnp.array(biases, dtype=jnp.float32),
            jnp.array(J, dtype=jnp.float32),
            beta=5.0,
        )

        samples_np = np.array(samples)
        test_inputs = []
        for i in range(samples_np.shape[0]):
            args = decode_spins(samples_np[i], encodings)
            # Add default values for non-fuzzable parameters.
            for name, kind in params:
                if name not in args:
                    if kind == "str":
                        args[name] = "test"
                    elif kind == "list":
                        args[name] = [1, 2, 3]
                    else:
                        args[name] = 1
            test_inputs.append(args)

        result = differential_fuzz(buggy_func, ref_func, test_inputs)
        return {
            "bugs_found": result.bugs_found,
            "unique_sigs": result.unique_sigs,
            "fuzz_inputs": result.total_inputs,
            "errors": [(e[0], e[1], e[2]) for e in result.errors[:5]],
        }
    except Exception as e:
        return {
            "bugs_found": 0, "unique_sigs": set(), "fuzz_inputs": 0,
            "errors": [f"Ising fuzz failed: {e}"],
        }


def _get_param_types(code: str, entry_point: str) -> list[tuple[str, str]]:
    """Extract parameter types from a function definition in code.

    **Detailed explanation for engineers:**
        Parses the AST to find the entry_point function and reads its parameter
        annotations. Maps Python type names to the Ising encoding types used by
        Exp 54: 'int' -> 'int', 'list' -> 'list_int', everything else -> as-is.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            params: list[tuple[str, str]] = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                kind = "int"  # default
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    ann = arg.annotation.id
                    if ann == "list":
                        kind = "list_int"
                    elif ann in ("int", "float", "str", "bool"):
                        kind = ann
                params.append((arg.arg, kind))
            return params
    return []


# ---------------------------------------------------------------------------
# 5. Verify-repair loop for code problems
# ---------------------------------------------------------------------------

def verify_repair_code(
    problem: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    max_iters: int = 3,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Run verify-repair loop on a single HumanEval problem.

    **Detailed explanation for engineers:**
        For each problem, this function:

        1. Generates a code solution (LLM or simulated).
        2. Runs canonical test cases.
        3. Extracts constraints via AST (Exp 48).
        4. Instruments code for runtime checks (Exp 53).
        5. Runs Ising-guided fuzz tests (Exp 54).
        6. If any failures: builds a repair prompt with failure details and
           regenerates (up to max_iters times).

        Returns a dict with per-iteration traces and final pass/fail status.
    """
    if rng is None:
        rng = random.Random(68)

    entry = problem["entry_point"]
    prompt = problem["prompt"]
    canonical = problem["canonical_solution"]
    test_cases = problem["test_cases"]

    iterations: list[dict[str, Any]] = []

    for iteration in range(max_iters + 1):
        # Step 1: Generate or simulate a solution.
        if use_live_llm and tokenizer and model:
            if iteration == 0:
                gen_prompt = (
                    f"Write a Python function that satisfies this specification.\n"
                    f"Return ONLY the Python code, no explanation.\n\n"
                    f"{prompt}"
                )
            else:
                # Build repair prompt with failure feedback.
                prev = iterations[-1]
                feedback_lines = ["Your previous code had these issues:"]
                for err in prev.get("test_errors", [])[:5]:
                    feedback_lines.append(f"  - {err}")
                for err in prev.get("fuzz_errors", [])[:3]:
                    feedback_lines.append(f"  - Fuzz found: {err}")
                for viol in prev.get("instrumentation_violations", [])[:3]:
                    feedback_lines.append(f"  - Runtime check: {viol}")
                feedback = "\n".join(feedback_lines)
                gen_prompt = (
                    f"Write a Python function that satisfies this specification.\n"
                    f"Return ONLY the Python code, no explanation.\n\n"
                    f"{prompt}\n\n"
                    f"Your previous attempt:\n{prev['code']}\n\n"
                    f"{feedback}\n\n"
                    f"Please fix these issues and provide corrected code."
                )
            response = generate_with_llm(gen_prompt, tokenizer, model, device)
            code = extract_code_from_response(response)
        else:
            code = simulate_solution(problem, iteration=iteration, rng=rng)

        # Step 2: Run canonical test cases.
        test_result = run_test_cases(code, entry, test_cases)

        # Step 3: Run instrumentation checks (Exp 48 + 53).
        instr_result = run_instrumentation(code, entry)

        # Step 4: Run Ising-guided fuzzing (Exp 54).
        fuzz_result = run_ising_fuzz(
            code, entry, canonical, prompt, n_fuzz=20,
        )

        # Collect this iteration's results.
        iter_data = {
            "iteration": iteration,
            "code": code,
            "test_passed": test_result["passed"],
            "test_failed": test_result["failed"],
            "test_all_pass": test_result["all_pass"],
            "test_errors": test_result["errors"],
            "instrumentation_detected": instr_result["detected"],
            "instrumentation_violations": instr_result.get("dynamic_violations", []),
            "n_static_violations": instr_result["n_static_violations"],
            "n_dynamic_violations": instr_result["n_dynamic_violations"],
            "fuzz_bugs_found": fuzz_result["bugs_found"],
            "fuzz_unique_sigs": fuzz_result["unique_sigs"],
            "fuzz_inputs": fuzz_result["fuzz_inputs"],
            "fuzz_errors": fuzz_result.get("errors", []),
        }
        iterations.append(iter_data)

        # All good? Stop early.
        # Note: static-only violations (from Exp 48's AST analysis) have a
        # known false-positive rate for comprehension variables (e.g., 'c' in
        # a generator expression gets flagged as "uninitialized"). We only
        # block on dynamic violations (runtime instrumentation failures) and
        # actual test/fuzz failures, not static-only constraint violations.
        all_pass = (
            test_result["all_pass"]
            and instr_result["n_dynamic_violations"] == 0
            and fuzz_result["bugs_found"] == 0
        )
        if all_pass:
            break

    # Summarize results.
    first_iter = iterations[0]
    last_iter = iterations[-1]
    initial_pass = (
        first_iter["test_all_pass"]
        and first_iter["n_dynamic_violations"] == 0
        and first_iter["fuzz_bugs_found"] == 0
    )
    final_pass = (
        last_iter["test_all_pass"]
        and last_iter["n_dynamic_violations"] == 0
        and last_iter["fuzz_bugs_found"] == 0
    )

    # Determine which detection methods found unique bugs.
    bugs_by_tests = first_iter["test_failed"] > 0
    bugs_by_instrumentation = first_iter["instrumentation_detected"]
    bugs_by_fuzzing = first_iter["fuzz_bugs_found"] > 0

    # A bug is "fuzzing-only" if tests passed but fuzzing found disagreements.
    fuzzing_only = (
        first_iter["test_all_pass"]
        and first_iter["fuzz_bugs_found"] > 0
    )

    return {
        "task_id": problem["task_id"],
        "entry_point": entry,
        "initial_pass": initial_pass,
        "final_pass": final_pass,
        "n_repairs": len(iterations) - 1,
        "repaired": not initial_pass and final_pass,
        "bugs_by_tests": bugs_by_tests,
        "bugs_by_instrumentation": bugs_by_instrumentation,
        "bugs_by_fuzzing": bugs_by_fuzzing,
        "fuzzing_only_bugs": fuzzing_only,
        "fuzz_unique_sigs": first_iter["fuzz_unique_sigs"],
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# 6. Main benchmark
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the HumanEval-style benchmark: 50 problems, full verification pipeline."""
    print("=" * 78)
    print("EXPERIMENT 68: HumanEval Code Verification Benchmark")
    print("  50 problems × (generate + constraints + instrument + fuzz + repair)")
    print("  Pipeline: Exp 48 (constraints) + Exp 53 (instrument) + Exp 54 (fuzz)")
    print("=" * 78)

    overall_start = time.time()

    # --- Load problems ---
    print("\n  Loading HumanEval problems...")
    problems = load_humaneval_problems()
    print(f"  {len(problems)} problems loaded.")

    # --- Load LLM ---
    print("\n  Attempting to load LLM...")
    tokenizer, model, device, use_live_llm = load_llm()

    if not use_live_llm:
        print("\n  *** FALLBACK: Using simulated code solutions ***")
        print("  (Model loading failed — pipeline logic is still exercised)")
        print("  Simulated: ~70% correct on first attempt, ~30% buggy")

    # --- Run benchmark ---
    results: list[dict[str, Any]] = []

    for i, problem in enumerate(problems):
        rng = random.Random(68_000 + i)
        result = verify_repair_code(
            problem,
            tokenizer=tokenizer,
            model=model,
            device=device,
            use_live_llm=use_live_llm,
            max_iters=3,
            rng=rng,
        )
        results.append(result)

        # Progress indicator every 10 problems.
        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r["initial_pass"])
            print(f"    {i + 1}/{len(problems)} done "
                  f"(pass@1 so far: {n_pass}/{len(results)})")

    # --- Free LLM memory ---
    if use_live_llm:
        del model, tokenizer
        try:
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    # --- Compute metrics ---
    elapsed = time.time() - overall_start
    n_total = len(results)

    n_pass1 = sum(1 for r in results if r["initial_pass"])
    n_pass1_repair = sum(1 for r in results if r["final_pass"])
    n_repaired = sum(1 for r in results if r["repaired"])

    n_bugs_tests = sum(1 for r in results if r["bugs_by_tests"])
    n_bugs_instr = sum(1 for r in results if r["bugs_by_instrumentation"])
    n_bugs_fuzz = sum(1 for r in results if r["bugs_by_fuzzing"])
    n_fuzzing_only = sum(1 for r in results if r["fuzzing_only_bugs"])

    total_fuzz_sigs: set = set()
    for r in results:
        total_fuzz_sigs |= r.get("fuzz_unique_sigs", set())

    avg_repairs = (
        np.mean([r["n_repairs"] for r in results if r["n_repairs"] > 0])
        if any(r["n_repairs"] > 0 for r in results) else 0
    )

    # --- Display results ---
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"EXPERIMENT 68 RESULTS ({elapsed:.1f}s) "
          f"[{'LIVE LLM' if use_live_llm else 'SIMULATED'}]")
    print(sep)

    print(f"\n  Problems tested:             {n_total}")
    print(f"\n  === Pass Rates ===")
    print(f"  pass@1 (first generation):   {n_pass1}/{n_total} "
          f"({n_pass1 / n_total:.1%})")
    print(f"  pass@1+repair (max 3 iters): {n_pass1_repair}/{n_total} "
          f"({n_pass1_repair / n_total:.1%})")
    print(f"  Improvement from repair:     +{n_repaired} problems "
          f"(+{n_repaired / n_total:.1%})")

    print(f"\n  === Bug Detection Source Breakdown ===")
    print(f"  Bugs caught by test cases:     {n_bugs_tests}/{n_total}")
    print(f"  Bugs caught by instrumentation:{n_bugs_instr}/{n_total}")
    print(f"  Bugs caught by Ising fuzzing:  {n_bugs_fuzz}/{n_total}")
    print(f"  Unique bugs ONLY found by fuzzing (tests passed): {n_fuzzing_only}")
    print(f"  Total unique fuzz bug signatures: {len(total_fuzz_sigs)}")

    if avg_repairs > 0:
        print(f"\n  === Repair Statistics ===")
        n_needed = sum(1 for r in results if r["n_repairs"] > 0)
        print(f"  Problems needing repair:       {n_needed}/{n_total}")
        print(f"  Average repair iterations:     {avg_repairs:.1f}")
        print(f"  Repairs successful:            {n_repaired}/{n_needed}")

    # --- Per-problem detail for failures ---
    failures = [r for r in results if not r["final_pass"]]
    if failures:
        print(f"\n  === Remaining Failures ({len(failures)}) ===")
        for f in failures[:10]:
            last = f["iterations"][-1]
            print(f"    {f['task_id']} ({f['entry_point']}): "
                  f"tests={last['test_failed']} fail, "
                  f"fuzz={last['fuzz_bugs_found']} bugs, "
                  f"iters={f['n_repairs']}")

    # --- Verdict ---
    print(f"\n  {sep}")
    if n_pass1_repair > n_pass1:
        print(f"  VERDICT: Verify-repair loop improved pass rate from "
              f"{n_pass1 / n_total:.1%} to {n_pass1_repair / n_total:.1%}")
        print(f"  Ising-guided fuzzing found {n_fuzzing_only} bugs that "
              f"canonical tests missed.")
    elif n_pass1 == n_total:
        print(f"  VERDICT: All {n_total} problems passed on first generation "
              f"— no repair needed.")
    else:
        print(f"  VERDICT: pass@1={n_pass1 / n_total:.1%}, "
              f"pass@1+repair={n_pass1_repair / n_total:.1%}")

    print(f"\n  Architecture: Generate -> Constraint Extract (Exp 48) -> "
          f"Instrument (Exp 53)")
    print(f"               -> Test Cases -> Ising Fuzz (Exp 54) -> "
          f"Repair Loop (Exp 57)")
    print(f"  Constraint layer is deterministic — no hallucination in "
          f"verification.")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
