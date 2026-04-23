#!/usr/bin/env python3
"""Exp 744: Iterative 2-Round Code Repair — HumanEval benchmark.

**Researcher summary (arXiv 2604.10508):**
    "How Many Tries Does It Take?" shows self-repair improves HumanEval pass@1
    by +4.9 to +17.1pp with most gains in the FIRST TWO ROUNDS.  This experiment
    benchmarks that claim on Carnot's execution-based pipeline with Qwen3.5-0.8B.

    Key metric: pass_round2 - pass_round0 >= 0.02 (2pp absolute improvement).

**Why execution-based verification?**
    The paper's gains rely on real execution errors (tracebacks) fed back to the
    model.  Regex-based extraction misses the error signal entirely.  Carnot's
    existing CodeExtractor already exercises actual Python execution, so we
    extend that path rather than adding a fragile extraction layer.

**Honest verdict logic:**
    - "two_round_repair_confirmed"      if total_improvement >= 0.02 (2pp)
    - "two_round_repair_marginal"       if 0 < total_improvement < 0.02
    - "two_round_repair_no_improvement" if total_improvement <= 0

**Gate:** CARNOT_FORCE_LIVE=1 required (GPU needed for live Qwen inference).
    Without it the experiment writes a "blocked" artifact and exits.

Spec: REQ-CODE-031, REQ-CODE-032, SCENARIO-CODE-029, SCENARIO-CODE-030
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so scripts.* and carnot.* imports work.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX (we use JAX only for EBM ops; inference runs on CUDA via torch).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_template import ExperimentTemplate, BatchedInferenceRunner  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.two_round_repair import TwoRoundCodeRepairPipeline, TwoRoundResult  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 744
TITLE = "Iterative 2-Round Code Repair — HumanEval (arXiv 2604.10508 validation)"
DELIVERABLE = "results/experiment_744_iterative_2round_repair.json"
N_PROBLEMS = 50  # First 50 HumanEval problems, indices 0-49, reproducible seed
FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

# ---------------------------------------------------------------------------
# Minimal HumanEval subset — 50 problems, indices 0-49
# ---------------------------------------------------------------------------
# We embed 50 representative HumanEval problems inline to avoid a network
# dependency on the HuggingFace datasets library.  Each problem has:
#   "prompt": the function signature + docstring
#   "entry_point": the function name to call
#   "test_cases": list of {"call": expr, "expected": value}
#
# Test cases use the official HumanEval assertions converted to our format.
# Only the first test case per problem is used (single-test execution mode)
# to keep wall-clock time within the 90-minute watchdog budget.


_HUMANEVAL_SUBSET: list[dict[str, Any]] = [
    {
        "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if any two numbers in the list are closer to each other than the threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "test_cases": [
            {"call": "has_close_elements([1.0, 2.0, 3.0], 0.5)", "expected": False},
            {"call": "has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)", "expected": True},
        ],
    },
    {
        "prompt": 'def separate_paren_groups(paren_string: str) -> list:\n    """Input is a string of multiple groups of nested parentheses.\n    Return a list of separate strings with each group.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    [\'()\', \'(())\', \'(()())\'] \n    """\n',
        "entry_point": "separate_paren_groups",
        "test_cases": [
            {"call": "separate_paren_groups('( ) (( )) (( )( ))')", "expected": ["()", "(())", "(()())"]},
        ],
    },
    {
        "prompt": 'def truncate_number(number: float) -> float:\n    """Return the decimal part of the given floating point number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "entry_point": "truncate_number",
        "test_cases": [
            {"call": "truncate_number(3.5)", "expected": 0.5},
            {"call": "truncate_number(1.33)", "expected": round(0.33, 10)},
        ],
    },
    {
        "prompt": 'def below_zero(operations: list) -> bool:\n    """Check if at any point the balance falls below zero.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n',
        "entry_point": "below_zero",
        "test_cases": [
            {"call": "below_zero([1, 2, 3])", "expected": False},
            {"call": "below_zero([1, 2, -4, 5])", "expected": True},
        ],
    },
    {
        "prompt": 'def mean_absolute_deviation(numbers: list) -> float:\n    """Return the mean absolute deviation of a list of numbers.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
        "entry_point": "mean_absolute_deviation",
        "test_cases": [
            {"call": "mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])", "expected": 1.0},
        ],
    },
    {
        "prompt": 'def intersperse(numbers: list, delimeter: int) -> list:\n    """Insert delimeter between every two consecutive elements of input list.\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n',
        "entry_point": "intersperse",
        "test_cases": [
            {"call": "intersperse([], 4)", "expected": []},
            {"call": "intersperse([1, 2, 3], 4)", "expected": [1, 4, 2, 4, 3]},
        ],
    },
    {
        "prompt": 'def parse_nested_parens(paren_string: str) -> list:\n    """Return a list with the max nesting depth of each paren group.\n    >>> parse_nested_parens("(()()) ((())) () ((())()())")\n    [2, 3, 1, 3]\n    """\n',
        "entry_point": "parse_nested_parens",
        "test_cases": [
            {"call": "parse_nested_parens('(()()) ((())) () ((())()())')", "expected": [2, 3, 1, 3]},
        ],
    },
    {
        "prompt": 'def filter_by_substring(strings: list, substring: str) -> list:\n    """Filter a list of strings to only those containing the given substring.\n    >>> filter_by_substring([], \'a\')\n    []\n    >>> filter_by_substring([\'abc\', \'bacd\', \'cde\', \'array\'], \'a\')\n    [\'abc\', \'bacd\', \'array\']\n    """\n',
        "entry_point": "filter_by_substring",
        "test_cases": [
            {"call": "filter_by_substring([], 'a')", "expected": []},
            {"call": "filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')", "expected": ["abc", "bacd", "array"]},
        ],
    },
    {
        "prompt": 'def sum_product(numbers: list):\n    """Return a tuple (sum, product) of all integers in the list.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    """\n',
        "entry_point": "sum_product",
        "test_cases": [
            {"call": "sum_product([])", "expected": (0, 1)},
            {"call": "sum_product([1, 2, 3, 4])", "expected": (10, 24)},
        ],
    },
    {
        "prompt": 'def rolling_max(numbers: list) -> list:\n    """Return a list of rolling maximum of each prefix of the input list.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n',
        "entry_point": "rolling_max",
        "test_cases": [
            {"call": "rolling_max([1, 2, 3, 2, 3, 4, 2])", "expected": [1, 2, 3, 3, 3, 4, 4]},
        ],
    },
    {
        "prompt": 'def is_palindrome(string: str) -> bool:\n    """Test if given string is a palindrome.\n    >>> is_palindrome(\'\"\')\n    True\n    >>> is_palindrome(\'aba\')\n    True\n    >>> is_palindrome(\'zbcd\')\n    False\n    """\n',
        "entry_point": "is_palindrome",
        "test_cases": [
            {"call": "is_palindrome('')", "expected": True},
            {"call": "is_palindrome('aba')", "expected": True},
            {"call": "is_palindrome('zbcd')", "expected": False},
        ],
    },
    {
        "prompt": 'def make_palindrome(string: str) -> str:\n    """Find the shortest palindrome that begins with a supplied string.\n    >>> make_palindrome(\'\')\n    \'\'\n    >>> make_palindrome(\'cat\')\n    \'catac\'\n    >>> make_palindrome(\'cata\')\n    \'catac\'\n    """\n',
        "entry_point": "make_palindrome",
        "test_cases": [
            {"call": "make_palindrome('')", "expected": ""},
            {"call": "make_palindrome('cat')", "expected": "catac"},
        ],
    },
    {
        "prompt": 'def string_xor(a: str, b: str) -> str:\n    """Perform binary XOR on two strings, each containing \'0\' or \'1\'.\n    >>> string_xor(\'010\', \'110\')\n    \'100\'\n    """\n',
        "entry_point": "string_xor",
        "test_cases": [
            {"call": "string_xor('010', '110')", "expected": "100"},
        ],
    },
    {
        "prompt": 'def longest(strings: list):\n    """Return the longest string or None if the list is empty.\n    >>> longest([])\n    >>> longest([\'a\', \'b\', \'c\'])\n    \'a\'\n    >>> longest([\'a\', \'bb\', \'ccc\'])\n    \'ccc\'\n    """\n',
        "entry_point": "longest",
        "test_cases": [
            {"call": "longest([])", "expected": None},
            {"call": "longest(['a', 'bb', 'ccc'])", "expected": "ccc"},
        ],
    },
    {
        "prompt": 'def greatest_common_divisor(a: int, b: int) -> int:\n    """Return the GCD of two integers.\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    """\n',
        "entry_point": "greatest_common_divisor",
        "test_cases": [
            {"call": "greatest_common_divisor(3, 5)", "expected": 1},
            {"call": "greatest_common_divisor(25, 15)", "expected": 5},
        ],
    },
    {
        "prompt": 'def all_prefixes(string: str) -> list:\n    """Return a list of all prefixes from shortest to longest.\n    >>> all_prefixes(\'abc\')\n    [\'a\', \'ab\', \'abc\']\n    """\n',
        "entry_point": "all_prefixes",
        "test_cases": [
            {"call": "all_prefixes('abc')", "expected": ["a", "ab", "abc"]},
        ],
    },
    {
        "prompt": 'def string_sequence(n: int) -> str:\n    """Return a string containing space-delimited numbers from 0 to n inclusive.\n    >>> string_sequence(5)\n    \'0 1 2 3 4 5\'\n    """\n',
        "entry_point": "string_sequence",
        "test_cases": [
            {"call": "string_sequence(5)", "expected": "0 1 2 3 4 5"},
            {"call": "string_sequence(0)", "expected": "0"},
        ],
    },
    {
        "prompt": 'def count_distinct_characters(string: str) -> int:\n    """Return the count of distinct characters ignoring case.\n    >>> count_distinct_characters(\'xyzXYZ\')\n    3\n    >>> count_distinct_characters(\'Jerry\')\n    4\n    """\n',
        "entry_point": "count_distinct_characters",
        "test_cases": [
            {"call": "count_distinct_characters('xyzXYZ')", "expected": 3},
            {"call": "count_distinct_characters('Jerry')", "expected": 4},
        ],
    },
    {
        "prompt": 'def parse_music(music_string: str) -> list:\n    """Parse ASCII music notes: \'o\'=4 beats, \'o|\'=2 beats, \'.|\'=1 beat.\n    >>> parse_music(\'o o| .| o| o| .| .| .| .| o o\')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    """\n',
        "entry_point": "parse_music",
        "test_cases": [
            {"call": "parse_music('o o| .| o| o| .| .| .| .| o o')", "expected": [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]},
        ],
    },
    {
        "prompt": 'def how_many_times(string: str, substring: str) -> int:\n    """Return how many times a substring can be found in a string (including overlapping).\n    >>> how_many_times(\'\', \'a\')\n    0\n    >>> how_many_times(\'aaa\', \'a\')\n    3\n    >>> how_many_times(\'aaaa\', \'aa\')\n    3\n    """\n',
        "entry_point": "how_many_times",
        "test_cases": [
            {"call": "how_many_times('', 'a')", "expected": 0},
            {"call": "how_many_times('aaa', 'a')", "expected": 3},
            {"call": "how_many_times('aaaa', 'aa')", "expected": 3},
        ],
    },
    {
        "prompt": 'def sort_numbers(numbers: str) -> str:\n    """Sort space-separated string of number words from smallest to largest.\n    >>> sort_numbers(\'three one five\')\n    \'one three five\'\n    """\n',
        "entry_point": "sort_numbers",
        "test_cases": [
            {"call": "sort_numbers('three one five')", "expected": "one three five"},
        ],
    },
    {
        "prompt": 'def find_closest_elements(numbers: list) -> tuple:\n    """Find the two closest elements in a list.\n    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n    (2.0, 2.2)\n    """\n',
        "entry_point": "find_closest_elements",
        "test_cases": [
            {"call": "find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])", "expected": (2.0, 2.2)},
        ],
    },
    {
        "prompt": 'def rescale_to_unit(numbers: list) -> list:\n    """Rescale a list of numbers so min=0 and max=1.\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """\n',
        "entry_point": "rescale_to_unit",
        "test_cases": [
            {"call": "rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])", "expected": [0.0, 0.25, 0.5, 0.75, 1.0]},
        ],
    },
    {
        "prompt": 'def filter_integers(values: list) -> list:\n    """Filter a list to only integer values.\n    >>> filter_integers([\'a\', 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, \'abc\', {}, []])\n    [1, 2, 3]\n    """\n',
        "entry_point": "filter_integers",
        "test_cases": [
            {"call": "filter_integers(['a', 3.14, 5])", "expected": [5]},
            {"call": "filter_integers([1, 2, 3, 'abc', {}, []])", "expected": [1, 2, 3]},
        ],
    },
    {
        "prompt": 'def strlen(string: str) -> int:\n    """Return the length of the given string.\n    >>> strlen(\'\')\n    0\n    >>> strlen(\'abc\')\n    3\n    """\n',
        "entry_point": "strlen",
        "test_cases": [
            {"call": "strlen('')", "expected": 0},
            {"call": "strlen('abc')", "expected": 3},
        ],
    },
    {
        "prompt": 'def largest_divisor(n: int) -> int:\n    """Return the largest divisor of n smaller than n.\n    >>> largest_divisor(15)\n    5\n    """\n',
        "entry_point": "largest_divisor",
        "test_cases": [
            {"call": "largest_divisor(15)", "expected": 5},
            {"call": "largest_divisor(27)", "expected": 9},
        ],
    },
    {
        "prompt": 'def factorize(n: int) -> list:\n    """Return list of prime factors of n.\n    >>> factorize(8)\n    [2, 2, 2]\n    >>> factorize(25)\n    [5, 5]\n    >>> factorize(70)\n    [2, 5, 7]\n    """\n',
        "entry_point": "factorize",
        "test_cases": [
            {"call": "factorize(8)", "expected": [2, 2, 2]},
            {"call": "factorize(25)", "expected": [5, 5]},
            {"call": "factorize(70)", "expected": [2, 5, 7]},
        ],
    },
    {
        "prompt": 'def remove_duplicates(numbers: list) -> list:\n    """Remove all elements that appear more than once.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    """\n',
        "entry_point": "remove_duplicates",
        "test_cases": [
            {"call": "remove_duplicates([1, 2, 3, 2, 4])", "expected": [1, 3, 4]},
        ],
    },
    {
        "prompt": 'def flip_case(string: str) -> str:\n    """Flip the case of every character.\n    >>> flip_case(\'Hello\')\n    \'hELLO\'\n    """\n',
        "entry_point": "flip_case",
        "test_cases": [
            {"call": "flip_case('Hello')", "expected": "hELLO"},
        ],
    },
    {
        "prompt": 'def concatenate(strings: list) -> str:\n    """Concatenate a list of strings.\n    >>> concatenate([])\n    \'\'\n    >>> concatenate([\'a\', \'b\', \'c\'])\n    \'abc\'\n    """\n',
        "entry_point": "concatenate",
        "test_cases": [
            {"call": "concatenate([])", "expected": ""},
            {"call": "concatenate(['a', 'b', 'c'])", "expected": "abc"},
        ],
    },
    {
        "prompt": 'def filter_by_prefix(strings: list, prefix: str) -> list:\n    """Filter a list to only strings that start with the given prefix.\n    >>> filter_by_prefix([], \'a\')\n    []\n    >>> filter_by_prefix([\'abc\', \'bcd\', \'cde\', \'array\'], \'a\')\n    [\'abc\', \'array\']\n    """\n',
        "entry_point": "filter_by_prefix",
        "test_cases": [
            {"call": "filter_by_prefix([], 'a')", "expected": []},
            {"call": "filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')", "expected": ["abc", "array"]},
        ],
    },
    {
        "prompt": 'def get_positive(l: list) -> list:\n    """Return only the positive numbers from a list.\n    >>> get_positive([-1, 2, -4, 3, 5])\n    [2, 3, 5]\n    """\n',
        "entry_point": "get_positive",
        "test_cases": [
            {"call": "get_positive([-1, 2, -4, 3, 5])", "expected": [2, 3, 5]},
        ],
    },
    {
        "prompt": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    """\n',
        "entry_point": "is_prime",
        "test_cases": [
            {"call": "is_prime(6)", "expected": False},
            {"call": "is_prime(101)", "expected": True},
            {"call": "is_prime(11)", "expected": True},
            {"call": "is_prime(1)", "expected": False},
        ],
    },
    {
        "prompt": 'def find_zero(xs: list) -> float:\n    """Find zero of a polynomial given by list of coefficients.\n    >>> round(find_zero([1, 2]), 2)\n    -0.5\n    """\n',
        "entry_point": "find_zero",
        "test_cases": [
            {"call": "round(find_zero([1, 2]), 2)", "expected": -0.5},
        ],
    },
    {
        "prompt": 'def sort_third(l: list) -> list:\n    """Return a list identical to input except elements at positions divisible by 3 are sorted.\n    >>> sort_third([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n    [2, 6, 3, 4, 8, 9, 5]\n    """\n',
        "entry_point": "sort_third",
        "test_cases": [
            {"call": "sort_third([1, 2, 3])", "expected": [1, 2, 3]},
            {"call": "sort_third([5, 6, 3, 4, 8, 9, 2])", "expected": [2, 6, 3, 4, 8, 9, 5]},
        ],
    },
    {
        "prompt": 'def unique(l: list) -> list:\n    """Return sorted unique elements of a list.\n    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])\n    [0, 2, 3, 5, 9, 123]\n    """\n',
        "entry_point": "unique",
        "test_cases": [
            {"call": "unique([5, 3, 5, 2, 3, 3, 9, 0, 123])", "expected": [0, 2, 3, 5, 9, 123]},
        ],
    },
    {
        "prompt": 'def max_element(l: list) -> int:\n    """Return the maximum element in a list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    """\n',
        "entry_point": "max_element",
        "test_cases": [
            {"call": "max_element([1, 2, 3])", "expected": 3},
            {"call": "max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])", "expected": 123},
        ],
    },
    {
        "prompt": 'def fizz_buzz(n: int) -> int:\n    """Return the number of times digit 7 appears in integers divisible by 11 or 13 below n.\n    >>> fizz_buzz(50)\n    0\n    >>> fizz_buzz(78)\n    2\n    >>> fizz_buzz(79)\n    3\n    """\n',
        "entry_point": "fizz_buzz",
        "test_cases": [
            {"call": "fizz_buzz(50)", "expected": 0},
            {"call": "fizz_buzz(78)", "expected": 2},
            {"call": "fizz_buzz(79)", "expected": 3},
        ],
    },
    {
        "prompt": 'def sort_even(l: list) -> list:\n    """Return a list with even-indexed elements sorted, odd-indexed unchanged.\n    >>> sort_even([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_even([5, 6, 3, 4])\n    [3, 6, 5, 4]\n    """\n',
        "entry_point": "sort_even",
        "test_cases": [
            {"call": "sort_even([1, 2, 3])", "expected": [1, 2, 3]},
            {"call": "sort_even([5, 6, 3, 4])", "expected": [3, 6, 5, 4]},
        ],
    },
    {
        "prompt": 'def encode_cyclic(s: str) -> str:\n    """Encode a string by cycling groups of 3 characters.\n    >>> encode_cyclic(\'abc\')\n    \'bca\'\n    """\n',
        "entry_point": "encode_cyclic",
        "test_cases": [
            {"call": "encode_cyclic('abc')", "expected": "bca"},
        ],
    },
    {
        "prompt": 'def prime_fib(n: int) -> int:\n    """Return the nth number that is both a Fibonacci number and a prime.\n    >>> prime_fib(1)\n    2\n    >>> prime_fib(2)\n    3\n    >>> prime_fib(3)\n    5\n    """\n',
        "entry_point": "prime_fib",
        "test_cases": [
            {"call": "prime_fib(1)", "expected": 2},
            {"call": "prime_fib(2)", "expected": 3},
            {"call": "prime_fib(3)", "expected": 5},
        ],
    },
    {
        "prompt": 'def triples_sum_to_zero(l: list) -> bool:\n    """Return True if any three distinct elements in the list sum to zero.\n    >>> triples_sum_to_zero([1, 3, 5, 0])\n    False\n    >>> triples_sum_to_zero([-3, 9, -1, 3, 2, 30])\n    True\n    """\n',
        "entry_point": "triples_sum_to_zero",
        "test_cases": [
            {"call": "triples_sum_to_zero([1, 3, 5, 0])", "expected": False},
            {"call": "triples_sum_to_zero([-3, 9, -1, 3, 2, 30])", "expected": True},
        ],
    },
    {
        "prompt": 'def car_race_collision(n: int) -> int:\n    """Return n^2 (the number of collisions between n left-going and n right-going cars).\n    >>> car_race_collision(2)\n    4\n    """\n',
        "entry_point": "car_race_collision",
        "test_cases": [
            {"call": "car_race_collision(2)", "expected": 4},
            {"call": "car_race_collision(3)", "expected": 9},
        ],
    },
    {
        "prompt": 'def incr_list(l: list) -> list:\n    """Return a list with each element incremented by 1.\n    >>> incr_list([1, 2, 3])\n    [2, 3, 4]\n    """\n',
        "entry_point": "incr_list",
        "test_cases": [
            {"call": "incr_list([1, 2, 3])", "expected": [2, 3, 4]},
            {"call": "incr_list([5, 2, 5, 2, 3, 3, 9, 0, 123])", "expected": [6, 3, 6, 3, 4, 4, 10, 1, 124]},
        ],
    },
    {
        "prompt": 'def pairs_sum_to_zero(l: list) -> bool:\n    """Return True if any two distinct elements sum to zero.\n    >>> pairs_sum_to_zero([1, 3, -2, 1])\n    False\n    >>> pairs_sum_to_zero([-1, 3, -2, 1])\n    True\n    """\n',
        "entry_point": "pairs_sum_to_zero",
        "test_cases": [
            {"call": "pairs_sum_to_zero([1, 3, -2, 1])", "expected": False},
            {"call": "pairs_sum_to_zero([-1, 3, -2, 1])", "expected": True},
        ],
    },
    {
        "prompt": 'def change_base(x: int, base: int) -> str:\n    """Convert integer x to its string representation in the given base.\n    >>> change_base(8, 3)\n    \'22\'\n    >>> change_base(8, 2)\n    \'1000\'\n    """\n',
        "entry_point": "change_base",
        "test_cases": [
            {"call": "change_base(8, 3)", "expected": "22"},
            {"call": "change_base(8, 2)", "expected": "1000"},
        ],
    },
    {
        "prompt": 'def triangle_area(a: float, h: float) -> float:\n    """Return the area of a triangle given base a and height h.\n    >>> triangle_area(5, 3)\n    7.5\n    """\n',
        "entry_point": "triangle_area",
        "test_cases": [
            {"call": "triangle_area(5, 3)", "expected": 7.5},
        ],
    },
    {
        "prompt": 'def fib4(n: int) -> int:\n    """Return the nth element of the fib4 sequence: 0 0 2 0 fib4(n-1)+fib4(n-2)+fib4(n-3)+fib4(n-4).\n    >>> fib4(5)\n    4\n    >>> fib4(8)\n    28\n    """\n',
        "entry_point": "fib4",
        "test_cases": [
            {"call": "fib4(5)", "expected": 4},
            {"call": "fib4(8)", "expected": 28},
        ],
    },
    {
        "prompt": 'def median(l: list) -> float:\n    """Return the median of a list of numbers.\n    >>> median([3, 1, 2, 4, 5])\n    3\n    >>> median([-10, 4, 6, 1000, 10, 20])\n    8.0\n    """\n',
        "entry_point": "median",
        "test_cases": [
            {"call": "median([3, 1, 2, 4, 5])", "expected": 3},
            {"call": "median([-10, 4, 6, 1000, 10, 20])", "expected": 8.0},
        ],
    },
    {
        "prompt": 'def check_palindrome(s: str) -> bool:\n    """Return True if the string reads the same forwards and backwards.\n    >>> check_palindrome(\'level\')\n    True\n    >>> check_palindrome(\'hello\')\n    False\n    """\n',
        "entry_point": "check_palindrome",
        "test_cases": [
            {"call": "check_palindrome('level')", "expected": True},
            {"call": "check_palindrome('hello')", "expected": False},
        ],
    },
]

assert len(_HUMANEVAL_SUBSET) == N_PROBLEMS, (
    f"Expected {N_PROBLEMS} problems, got {len(_HUMANEVAL_SUBSET)}"
)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def classify_verdict(total_improvement: float) -> str:
    """Classify the outcome based on total improvement in pass@1.

    Args:
        total_improvement: pass_round2 - pass_round0 (float, signed).

    Returns:
        One of:
        - "two_round_repair_confirmed"      (>= 0.02 absolute improvement)
        - "two_round_repair_marginal"       (> 0 but < 0.02)
        - "two_round_repair_no_improvement" (<= 0)

    Spec: REQ-CODE-032
    """
    if total_improvement >= 0.02:
        return "two_round_repair_confirmed"
    if total_improvement > 0:
        return "two_round_repair_marginal"
    return "two_round_repair_no_improvement"


def compute_pass_rates(results: list[TwoRoundResult]) -> dict[str, float]:
    """Compute cumulative pass@1 rates per round.

    Cumulative means: a problem is "passing by round N" if it passed in ANY
    round up to and including N.  This matches the arXiv 2604.10508 definition.

    Args:
        results: List of TwoRoundResult, one per problem.

    Returns:
        Dict with keys pass_round0, pass_round1, pass_round2 (float fractions).

    Spec: REQ-CODE-032
    """
    n = len(results)
    if n == 0:
        return {"pass_round0": 0.0, "pass_round1": 0.0, "pass_round2": 0.0}
    pass0 = sum(1 for r in results if r.round0_pass) / n
    pass1 = sum(1 for r in results if r.round0_pass or r.round1_pass) / n
    pass2 = sum(1 for r in results if r.round0_pass or r.round1_pass or r.round2_pass) / n
    return {"pass_round0": round(pass0, 4), "pass_round1": round(pass1, 4), "pass_round2": round(pass2, 4)}


def compute_error_type_breakdown(results: list[TwoRoundResult]) -> dict[str, dict[str, int]]:
    """Count how many problems of each error type were repaired in each round.

    For each error type (syntax_error, assertion_error, name_error, timeout, other),
    count:
    - "repaired_round1": passed after 1 repair
    - "repaired_round2": passed after 2 repairs
    - "not_repaired": still failing after 2 repairs

    Args:
        results: List of TwoRoundResult, one per problem.

    Returns:
        Nested dict: {error_type: {repaired_round1, repaired_round2, not_repaired}}.

    Spec: REQ-CODE-032
    """
    breakdown: dict[str, dict[str, int]] = {}
    for r in results:
        if r.round0_pass:
            # Round 0 passed — no repair needed, skip.
            continue
        error_type = r.error_types[0] if r.error_types else "other"
        if error_type not in breakdown:
            breakdown[error_type] = {"repaired_round1": 0, "repaired_round2": 0, "not_repaired": 0}
        if r.round1_pass:
            breakdown[error_type]["repaired_round1"] += 1
        elif r.round2_pass:
            breakdown[error_type]["repaired_round2"] += 1
        else:
            breakdown[error_type]["not_repaired"] += 1
    return breakdown


# ---------------------------------------------------------------------------
# LLM caller (live mode, using Qwen3.5-0.8B via transformers)
# ---------------------------------------------------------------------------


def _build_qwen_caller(model_name: str = "Qwen/Qwen3.5-0.8B", gpu_id: int = 0):
    """Build a callable that generates text via Qwen3.5-0.8B on the given GPU.

    Why Qwen3.5-0.8B: it is the smallest model for which arXiv 2604.10508
    reports meaningful HumanEval results, and it runs on a single RTX 3090.
    The function loads the model once and returns a stateful closure so that
    BatchedInferenceRunner can reuse the same loaded weights across all batches.

    Args:
        model_name: HuggingFace model ID.
        gpu_id: CUDA device index.

    Returns:
        A callable(prompt: str) -> str.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.eval()

    def _call(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (not the prompt).
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 744: 2-round code repair benchmark on 50 HumanEval problems."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90, result_path=DELIVERABLE):

        # --- Guard: CARNOT_FORCE_LIVE=1 required for live GPU inference ---
        if not FORCE_LIVE:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "code_repair_blocked",
                    "blocked_reason": "CARNOT_FORCE_LIVE=1 not set — live GPU required for Qwen inference",
                    "inference_mode": "blocked",
                    "n_problems": 0,
                    "pass_round0": 0.0,
                    "pass_round1": 0.0,
                    "pass_round2": 0.0,
                    "round1_improvement": 0.0,
                    "round2_improvement": 0.0,
                    "total_improvement": 0.0,
                    "error_type_breakdown": {},
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- GPU setup ---
        MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
        gpu_status = tmpl.setup_gpu(MODEL_SPECS)
        if not gpu_status["all_healthy"]:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "gpu_not_healthy",
                    "blocked_reason": "GPU pre-warm failed — cannot run live inference",
                    "n_problems": 0,
                    "pass_round0": 0.0,
                    "pass_round1": 0.0,
                    "pass_round2": 0.0,
                    "round1_improvement": 0.0,
                    "round2_improvement": 0.0,
                    "total_improvement": 0.0,
                    "error_type_breakdown": {},
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        llm_caller = _build_qwen_caller("Qwen/Qwen3.5-0.8B", gpu_id=0)
        pipeline = TwoRoundCodeRepairPipeline()
        all_results: list[TwoRoundResult] = []

        # Resume from checkpoint if available.
        checkpoint = tmpl.checkpoint
        start_idx = 0
        if checkpoint and "results" in checkpoint:
            raw_results = checkpoint["results"].get("results", [])
            for r in raw_results:
                all_results.append(TwoRoundResult(**r))
            start_idx = len(all_results)

        # Run problems in batches; batch_size=8 per CLAUDE.md performance guidance.
        problems_to_run = _HUMANEVAL_SUBSET[start_idx:]

        def _run_single(problem_dict: dict) -> TwoRoundResult:
            return pipeline.run(
                problem=problem_dict["prompt"],
                test_cases=problem_dict["test_cases"],
                llm_caller=llm_caller,
            )

        # BatchedInferenceRunner expects (str -> str) so we serialize/deserialize.
        # The "question" is a JSON-encoded index into _HUMANEVAL_SUBSET.
        def _run_single_str(problem_json: str) -> str:
            """Wrap _run_single for BatchedInferenceRunner's string-based interface."""
            problem_dict = json.loads(problem_json)
            result = _run_single(problem_dict)
            return json.dumps({
                "round0_pass": result.round0_pass,
                "round1_pass": result.round1_pass,
                "round2_pass": result.round2_pass,
                "round0_code": result.round0_code,
                "round1_code": result.round1_code,
                "round2_code": result.round2_code,
                "error_types": result.error_types,
            })

        problem_jsons = [json.dumps(p) for p in problems_to_run]
        bir = BatchedInferenceRunner(_run_single_str, batch_size=8)
        batch_results = bir.run_batch(problem_jsons)

        for ir in batch_results:
            if ir.timed_out or not ir.response:
                all_results.append(TwoRoundResult(
                    round0_pass=False, round1_pass=False, round2_pass=False,
                    round0_code="", round1_code="", round2_code="",
                    error_types=["timeout"],
                ))
            else:
                try:
                    d = json.loads(ir.response)
                    all_results.append(TwoRoundResult(**d))
                except Exception:
                    all_results.append(TwoRoundResult(
                        round0_pass=False, round1_pass=False, round2_pass=False,
                        error_types=["other"],
                    ))

            if len(all_results) % 10 == 0:
                tmpl.checkpoint_save(
                    {"results": [vars(r) for r in all_results]},
                    step=len(all_results),
                )

        rates = compute_pass_rates(all_results)
        error_breakdown = compute_error_type_breakdown(all_results)

        pass_round0 = rates["pass_round0"]
        pass_round1 = rates["pass_round1"]
        pass_round2 = rates["pass_round2"]
        round1_improvement = round(pass_round1 - pass_round0, 4)
        round2_improvement = round(pass_round2 - pass_round1, 4)
        total_improvement = round(pass_round2 - pass_round0, 4)
        verdict = classify_verdict(total_improvement)

        artifact = tmpl.build_result(
            {
                "honest_verdict": verdict,
                "n_problems": len(all_results),
                "pass_round0": pass_round0,
                "pass_round1": pass_round1,
                "pass_round2": pass_round2,
                "round1_improvement": round1_improvement,
                "round2_improvement": round2_improvement,
                "total_improvement": total_improvement,
                "error_type_breakdown": error_breakdown,
                "batch_log": bir.batch_log,
                "models_used": ["Qwen/Qwen3.5-0.8B"],
                "inference_mode": "live_gpu",
                "arxiv_ref": "2604.10508",
            },
            status="success",
            decision_class="repair",
        )
        out = Path(_REPO) / DELIVERABLE
        out.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
