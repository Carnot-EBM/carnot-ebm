"""Exp 2990 verifier-backed hard-code stress manifest.

Spec: REQ-CODE-2990, SCENARIO-CODE-2990.

The manifest is intentionally small and executable. Its job is not to claim a
new repair result; it gives the next repair rerun a hard, replayable gate where
every retained item has one broken baseline candidate and one passing reference
solution under the same deterministic assertion suite.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_FILENAME = "experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json"
SCHEMA = "carnot.verifier_backed_hard_code_stress_manifest.v1"
RUN_DATE = "20260524"
DEFAULT_MANIFEST_REL_PATH = Path("datasets/repair_hard/manifest_v1.jsonl")
DEFAULT_TRANSCRIPT_REL_PATH = Path(
    "results/verifier_transcripts/experiment_2990/hard_code_stress_transcript_v1.jsonl"
)
INFERENCE_SUBSTRATE = "deterministic_executable_manifest_generation"
VALIDATION_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_2990_hard_code_stress_manifest.py -q",
    ".venv/bin/pytest tests/python -q",
    "python scripts/check_spec_coverage.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "hard_code_stress_set_ready",
    "manifest_path",
    "n_items",
    "all_items_have_tests",
    "all_baseline_candidates_fail",
    "all_reference_solutions_pass",
    "flaky_items",
    "verifier_transcript_paths",
    "hard_generation_sources",
    "honest_verdict",
)
NONDETERMINISTIC_TOKENS = ("random", "time.", "__import__('random')", '__import__("random")')


@dataclass(frozen=True)
class VerificationOutcome:
    """Execution evidence for one candidate under one item's assertion suite."""

    passed: bool
    candidate_key: str
    candidate_sha256: str
    test_suite_sha256: str
    tests_run: int
    failing_test_ids: list[str]
    errors: list[JsonDict]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def as_dict(self) -> JsonDict:
        return {
            "passed": self.passed,
            "candidate_key": self.candidate_key,
            "candidate_sha256": self.candidate_sha256,
            "test_suite_sha256": self.test_suite_sha256,
            "tests_run": self.tests_run,
            "failing_test_ids": list(self.failing_test_ids),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock injection for Exp 2990."""

    repo_root: Path = REPO_ROOT
    manifest_path: Path | None = None
    transcript_path: Path | None = None
    output_path: Path | None = None
    manifest_items: Sequence[JsonDict] | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFunc = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / DEFAULT_MANIFEST_REL_PATH

    def resolved_transcript_path(self) -> Path:
        return self.transcript_path or self.repo_root / DEFAULT_TRANSCRIPT_REL_PATH

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_no}: invalid JSONL row") from exc
    return rows


def load_manifest(path: Path) -> list[JsonDict]:
    return load_jsonl(path)


def default_items() -> tuple[JsonDict, ...]:
    """Return the deterministic 24-item stress set used by Exp 2990."""

    return (
        _item(
            "repair-hard-0001",
            "clamp_score",
            "Clamp x into the inclusive [lo, hi] range.",
            "def clamp_score(x, lo, hi):\n    return min(x, hi)\n",
            "def clamp_score(x, lo, hi):\n    return max(lo, min(x, hi))\n",
            [
                "assert clamp_score(12, 0, 10) == 10",
                "assert clamp_score(-3, 0, 10) == 0",
                "assert clamp_score(5, 0, 10) == 5",
            ],
            "Lower-bound edge case catches candidates that only cap the upper bound.",
        ),
        _item(
            "repair-hard-0002",
            "unique_preserve_order",
            "Return each distinct input value once, preserving first-seen order.",
            "def unique_preserve_order(items):\n    return sorted(set(items))\n",
            (
                "def unique_preserve_order(items):\n"
                "    seen = set()\n"
                "    out = []\n"
                "    for item in items:\n"
                "        if item not in seen:\n"
                "            seen.add(item)\n"
                "            out.append(item)\n"
                "    return out\n"
            ),
            [
                "assert unique_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]",
                "assert unique_preserve_order([]) == []",
                "assert unique_preserve_order(['b', 'a', 'b']) == ['b', 'a']",
            ],
            "Order preservation distinguishes true repair from set-based deduplication.",
        ),
        _item(
            "repair-hard-0003",
            "count_vowels",
            "Count vowels in a string, treating uppercase and lowercase equally.",
            "def count_vowels(text):\n    return sum(1 for ch in text if ch in 'aeiou')\n",
            (
                "def count_vowels(text):\n"
                "    return sum(1 for ch in text.lower() if ch in 'aeiou')\n"
            ),
            [
                "assert count_vowels('Education') == 5",
                "assert count_vowels('SKY') == 0",
                "assert count_vowels('Queue') == 4",
            ],
            "Case-folded examples catch candidates that overfit lowercase fixtures.",
        ),
        _item(
            "repair-hard-0004",
            "median_sorted",
            "Return the median of an already sorted non-empty numeric list.",
            "def median_sorted(values):\n    return values[len(values) // 2]\n",
            (
                "def median_sorted(values):\n"
                "    mid = len(values) // 2\n"
                "    if len(values) % 2:\n"
                "        return values[mid]\n"
                "    return (values[mid - 1] + values[mid]) / 2\n"
            ),
            [
                "assert median_sorted([1, 2, 9]) == 2",
                "assert median_sorted([1, 2, 3, 4]) == 2.5",
                "assert median_sorted([-5, -1, 0, 9]) == -0.5",
            ],
            "Even-length medians expose an off-by-one candidate that passes odd cases.",
        ),
        _item(
            "repair-hard-0005",
            "flatten_once",
            "Flatten exactly one list/tuple nesting level.",
            (
                "def flatten_once(items):\n"
                "    out = []\n"
                "    for item in items:\n"
                "        if isinstance(item, (list, tuple)):\n"
                "            out.extend(flatten_once(item))\n"
                "        else:\n"
                "            out.append(item)\n"
                "    return out\n"
            ),
            (
                "def flatten_once(items):\n"
                "    out = []\n"
                "    for item in items:\n"
                "        if isinstance(item, (list, tuple)):\n"
                "            out.extend(item)\n"
                "        else:\n"
                "            out.append(item)\n"
                "    return out\n"
            ),
            [
                "assert flatten_once([1, [2, 3], (4, 5)]) == [1, 2, 3, 4, 5]",
                "assert flatten_once([[1, [2]], 3]) == [1, [2], 3]",
                "assert flatten_once([]) == []",
            ],
            "Exactly-one-level behavior catches overzealous recursive flattening.",
        ),
        _item(
            "repair-hard-0006",
            "is_palindrome_text",
            "Return True when alphanumeric characters form a case-insensitive palindrome.",
            "def is_palindrome_text(text):\n    return text == text[::-1]\n",
            (
                "def is_palindrome_text(text):\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
            ),
            [
                "assert is_palindrome_text('A man, a plan, a canal: Panama') is True",
                "assert is_palindrome_text('race a car') is False",
                "assert is_palindrome_text('') is True",
            ],
            "Punctuation and case normalization distinguish prompt intent from raw reversal.",
        ),
        _item(
            "repair-hard-0007",
            "parse_bool",
            "Parse common boolean strings and booleans into True or False.",
            "def parse_bool(value):\n    return bool(value)\n",
            (
                "def parse_bool(value):\n"
                "    if isinstance(value, bool):\n"
                "        return value\n"
                "    lowered = str(value).strip().lower()\n"
                "    if lowered in {'true', 'yes', '1', 'on'}:\n"
                "        return True\n"
                "    if lowered in {'false', 'no', '0', 'off'}:\n"
                "        return False\n"
                "    raise ValueError('unknown boolean value')\n"
            ),
            [
                "assert parse_bool('true') is True",
                "assert parse_bool(' False ') is False",
                "assert parse_bool(False) is False",
            ],
            "Truthy string behavior catches candidates that wrap inputs in bool().",
        ),
        _item(
            "repair-hard-0008",
            "safe_divide",
            "Divide a by b, returning default when b is zero.",
            "def safe_divide(a, b, default=None):\n    return a / b\n",
            "def safe_divide(a, b, default=None):\n    return default if b == 0 else a / b\n",
            [
                "assert safe_divide(8, 2) == 4",
                "assert safe_divide(8, 0, default='n/a') == 'n/a'",
                "assert safe_divide(-9, 3) == -3",
            ],
            "Zero-denominator behavior keeps exception-hardening in the verifier suite.",
        ),
        _item(
            "repair-hard-0009",
            "chunked",
            "Split a sequence into lists of size n, preserving a final short chunk.",
            "def chunked(seq, n):\n    return [list(seq[i:i+n]) for i in range(0, len(seq) - n + 1, n)]\n",
            "def chunked(seq, n):\n    return [list(seq[i:i+n]) for i in range(0, len(seq), n)]\n",
            [
                "assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
                "assert chunked([], 3) == []",
                "assert chunked('abcd', 3) == [['a', 'b', 'c'], ['d']]",
            ],
            "Remainder chunks catch exact-multiple-only repairs.",
        ),
        _item(
            "repair-hard-0010",
            "rotate_left",
            "Rotate a sequence left by k steps, with k allowed to exceed the length.",
            "def rotate_left(seq, k):\n    return list(seq[k:]) + list(seq[:k])\n",
            (
                "def rotate_left(seq, k):\n"
                "    seq = list(seq)\n"
                "    if not seq:\n"
                "        return []\n"
                "    k %= len(seq)\n"
                "    return seq[k:] + seq[:k]\n"
            ),
            [
                "assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]",
                "assert rotate_left([1, 2, 3], 5) == [3, 1, 2]",
                "assert rotate_left([], 10) == []",
            ],
            "Modulo rotation and empty input catch boundary arithmetic bugs.",
        ),
        _item(
            "repair-hard-0011",
            "merge_intervals",
            "Merge overlapping or touching closed intervals.",
            (
                "def merge_intervals(intervals):\n"
                "    intervals = sorted(intervals)\n"
                "    out = []\n"
                "    for start, end in intervals:\n"
                "        if not out or start >= out[-1][1]:\n"
                "            out.append([start, end])\n"
                "        else:\n"
                "            out[-1][1] = max(out[-1][1], end)\n"
                "    return out\n"
            ),
            (
                "def merge_intervals(intervals):\n"
                "    intervals = sorted(intervals)\n"
                "    out = []\n"
                "    for start, end in intervals:\n"
                "        if not out or start > out[-1][1]:\n"
                "            out.append([start, end])\n"
                "        else:\n"
                "            out[-1][1] = max(out[-1][1], end)\n"
                "    return out\n"
            ),
            [
                "assert merge_intervals([[1, 3], [2, 5], [8, 9]]) == [[1, 5], [8, 9]]",
                "assert merge_intervals([[1, 2], [2, 4]]) == [[1, 4]]",
                "assert merge_intervals([]) == []",
            ],
            "Touching interval semantics catch strict-overlap-only implementations.",
        ),
        _item(
            "repair-hard-0012",
            "normalize_whitespace",
            "Collapse any run of whitespace into a single space and trim the ends.",
            "def normalize_whitespace(text):\n    return text.strip()\n",
            "def normalize_whitespace(text):\n    return ' '.join(text.split())\n",
            [
                "assert normalize_whitespace('  alpha   beta\\n gamma  ') == 'alpha beta gamma'",
                "assert normalize_whitespace('\\t') == ''",
                "assert normalize_whitespace('one') == 'one'",
            ],
            "Internal whitespace collapse catches candidates that only trim boundaries.",
        ),
        _item(
            "repair-hard-0013",
            "first_non_none",
            "Return the first value that is not None, or default when none exists.",
            "def first_non_none(values, default=None):\n    return values[0] if values else default\n",
            (
                "def first_non_none(values, default=None):\n"
                "    for value in values:\n"
                "        if value is not None:\n"
                "            return value\n"
                "    return default\n"
            ),
            [
                "assert first_non_none([None, 0, 2], default=9) == 0",
                "assert first_non_none([None, None], default='x') == 'x'",
                "assert first_non_none(['a'], default='x') == 'a'",
            ],
            "Falsy-but-valid values distinguish None filtering from truthiness filtering.",
        ),
        _item(
            "repair-hard-0014",
            "top_k_counts",
            "Return (item, count) pairs sorted by count descending, then item ascending.",
            (
                "def top_k_counts(items, k):\n"
                "    counts = {}\n"
                "    for item in items:\n"
                "        counts[item] = counts.get(item, 0) + 1\n"
                "    return sorted(counts.items(), key=lambda pair: -pair[1])[:k]\n"
            ),
            (
                "def top_k_counts(items, k):\n"
                "    counts = {}\n"
                "    for item in items:\n"
                "        counts[item] = counts.get(item, 0) + 1\n"
                "    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k]\n"
            ),
            [
                "assert top_k_counts(['b', 'a', 'b', 'a', 'c'], 2) == [('a', 2), ('b', 2)]",
                "assert top_k_counts([], 3) == []",
                "assert top_k_counts(['x', 'x', 'y'], 1) == [('x', 2)]",
            ],
            "Tie ordering makes the oracle deterministic instead of insertion-order dependent.",
        ),
        _item(
            "repair-hard-0015",
            "binary_search_leftmost",
            "Return the leftmost index of target in a sorted list, or -1.",
        (
            "def binary_search_leftmost(nums, target):\n"
            "    found = -1\n"
            "    for index, value in enumerate(nums):\n"
            "        if value == target:\n"
            "            found = index\n"
            "    return found\n"
        ),
            (
                "def binary_search_leftmost(nums, target):\n"
                "    lo, hi = 0, len(nums)\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if nums[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid\n"
                "    return lo if lo < len(nums) and nums[lo] == target else -1\n"
            ),
            [
                "assert binary_search_leftmost([1, 2, 2, 2, 5], 2) == 1",
                "assert binary_search_leftmost([1, 3, 5], 4) == -1",
                "assert binary_search_leftmost([], 1) == -1",
            ],
            "Duplicate targets catch any-position search that misses leftmost semantics.",
        ),
        _item(
            "repair-hard-0016",
            "roman_to_int",
            "Convert a Roman numeral using subtractive pairs.",
            (
                "def roman_to_int(s):\n"
                "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
                "    return sum(values[ch] for ch in s)\n"
            ),
            (
                "def roman_to_int(s):\n"
                "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
                "    total = 0\n"
                "    for index, ch in enumerate(s):\n"
                "        value = values[ch]\n"
                "        if index + 1 < len(s) and value < values[s[index + 1]]:\n"
                "            total -= value\n"
                "        else:\n"
                "            total += value\n"
                "    return total\n"
            ),
            [
                "assert roman_to_int('III') == 3",
                "assert roman_to_int('IV') == 4",
                "assert roman_to_int('MCMXCIV') == 1994",
            ],
            "Subtractive notation catches additive-only partial repairs.",
        ),
        _item(
            "repair-hard-0017",
            "is_valid_parentheses",
            "Return True when parentheses are balanced and every prefix is valid.",
            "def is_valid_parentheses(s):\n    return s.count('(') == s.count(')')\n",
            (
                "def is_valid_parentheses(s):\n"
                "    balance = 0\n"
                "    for ch in s:\n"
                "        if ch == '(':\n"
                "            balance += 1\n"
                "        elif ch == ')':\n"
                "            balance -= 1\n"
                "            if balance < 0:\n"
                "                return False\n"
                "    return balance == 0\n"
            ),
            [
                "assert is_valid_parentheses('(())') is True",
                "assert is_valid_parentheses(')(') is False",
                "assert is_valid_parentheses('(()') is False",
            ],
            "Prefix validity catches count-only balance checks.",
        ),
        _item(
            "repair-hard-0018",
            "transpose",
            "Transpose a rectangular matrix and return rows as lists.",
            "def transpose(matrix):\n    return list(zip(*matrix))\n",
            "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
            [
                "assert transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]",
                "assert transpose([]) == []",
                "assert transpose([[7]]) == [[7]]",
            ],
            "List-vs-tuple output catches shape-only candidates.",
        ),
        _item(
            "repair-hard-0019",
            "fizzbuzz",
            "Return FizzBuzz labels for integers 1 through n inclusive.",
            (
                "def fizzbuzz(n):\n"
                "    out = []\n"
                "    for value in range(n):\n"
                "        if value % 15 == 0:\n"
                "            out.append('FizzBuzz')\n"
                "        elif value % 3 == 0:\n"
                "            out.append('Fizz')\n"
                "        elif value % 5 == 0:\n"
                "            out.append('Buzz')\n"
                "        else:\n"
                "            out.append(str(value))\n"
                "    return out\n"
            ),
            (
                "def fizzbuzz(n):\n"
                "    out = []\n"
                "    for value in range(1, n + 1):\n"
                "        if value % 15 == 0:\n"
                "            out.append('FizzBuzz')\n"
                "        elif value % 3 == 0:\n"
                "            out.append('Fizz')\n"
                "        elif value % 5 == 0:\n"
                "            out.append('Buzz')\n"
                "        else:\n"
                "            out.append(str(value))\n"
                "    return out\n"
            ),
            [
                "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
                "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
                "assert fizzbuzz(0) == []",
            ],
            "Inclusive range starts at one, catching zero-indexed FizzBuzz variants.",
        ),
        _item(
            "repair-hard-0020",
            "longest_common_prefix",
            "Return the longest common prefix for a list of strings.",
            "def longest_common_prefix(strings):\n    return strings[0] if strings else ''\n",
            (
                "def longest_common_prefix(strings):\n"
                "    if not strings:\n"
                "        return ''\n"
                "    prefix = strings[0]\n"
                "    for text in strings[1:]:\n"
                "        while not text.startswith(prefix):\n"
                "            prefix = prefix[:-1]\n"
                "            if not prefix:\n"
                "                return ''\n"
                "    return prefix\n"
            ),
            [
                "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'",
                "assert longest_common_prefix(['dog', 'racecar']) == ''",
                "assert longest_common_prefix([]) == ''",
            ],
            "Multi-string disagreement catches first-element shortcut candidates.",
        ),
        _item(
            "repair-hard-0021",
            "anagram_key",
            "Return a lowercase sorted alphanumeric key for anagram grouping.",
            "def anagram_key(text):\n    return ''.join(sorted(text))\n",
            (
                "def anagram_key(text):\n"
                "    chars = [ch.lower() for ch in text if ch.isalnum()]\n"
                "    return ''.join(sorted(chars))\n"
            ),
            [
                "assert anagram_key('Dormitory!!') == anagram_key('dirty room')",
                "assert anagram_key('A-b') == 'ab'",
                "assert anagram_key('') == ''",
            ],
            "Normalization catches raw-character sorting that leaks spaces and punctuation.",
        ),
        _item(
            "repair-hard-0022",
            "window_sums",
            "Return sums of every contiguous window of the requested size.",
            "def window_sums(nums, size):\n    return [sum(nums[i:i+size]) for i in range(len(nums) - size)]\n",
            (
                "def window_sums(nums, size):\n"
                "    if size <= 0 or size > len(nums):\n"
                "        return []\n"
                "    return [sum(nums[i:i+size]) for i in range(len(nums) - size + 1)]\n"
            ),
            [
                "assert window_sums([1, 2, 3, 4], 2) == [3, 5, 7]",
                "assert window_sums([5], 1) == [5]",
                "assert window_sums([1, 2], 3) == []",
            ],
            "Last-window and oversized-window cases catch off-by-one loop bounds.",
        ),
        _item(
            "repair-hard-0023",
            "parse_kv_pairs",
            "Parse semicolon-separated key=value pairs into a dictionary with trimmed fields.",
            (
                "def parse_kv_pairs(text):\n"
                "    out = {}\n"
                "    for part in text.split(','):\n"
                "        if '=' in part:\n"
                "            key, value = part.split('=', 1)\n"
                "            out[key] = value\n"
                "    return out\n"
            ),
            (
                "def parse_kv_pairs(text):\n"
                "    out = {}\n"
                "    for part in text.split(';'):\n"
                "        part = part.strip()\n"
                "        if not part:\n"
                "            continue\n"
                "        key, value = part.split('=', 1)\n"
                "        out[key.strip()] = value.strip()\n"
                "    return out\n"
            ),
            [
                "assert parse_kv_pairs('a=1; b = two ;') == {'a': '1', 'b': 'two'}",
                "assert parse_kv_pairs('') == {}",
                "assert parse_kv_pairs('x=1=2') == {'x': '1=2'}",
            ],
            "Delimiter, whitespace, and first-equals handling exercise parser edge cases.",
        ),
        _item(
            "repair-hard-0024",
            "grade_bucket",
            "Map numeric grades to A/B/C/D/F using inclusive lower bounds.",
            (
                "def grade_bucket(score):\n"
                "    if score > 90:\n"
                "        return 'A'\n"
                "    if score > 80:\n"
                "        return 'B'\n"
                "    if score > 70:\n"
                "        return 'C'\n"
                "    if score > 60:\n"
                "        return 'D'\n"
                "    return 'F'\n"
            ),
            (
                "def grade_bucket(score):\n"
                "    if score >= 90:\n"
                "        return 'A'\n"
                "    if score >= 80:\n"
                "        return 'B'\n"
                "    if score >= 70:\n"
                "        return 'C'\n"
                "    if score >= 60:\n"
                "        return 'D'\n"
                "    return 'F'\n"
            ),
            [
                "assert grade_bucket(90) == 'A'",
                "assert grade_bucket(80) == 'B'",
                "assert grade_bucket(59) == 'F'",
            ],
            "Boundary inclusivity catches strict-greater threshold candidates.",
        ),
    )


def run_candidate_tests(item: Mapping[str, Any], candidate_key: str) -> VerificationOutcome:
    tests = tuple(dict(row) for row in item.get("tests") or ())
    test_suite_sha = sha256_text(_canonical_json(tests))
    candidate = item.get(candidate_key)
    if not isinstance(candidate, str) or not candidate.strip():
        return _outcome(
            False,
            candidate_key,
            str(candidate or ""),
            test_suite_sha,
            0,
            [{"test_id": "candidate-load", "error_type": "missing_candidate", "message": candidate_key}],
        )
    if not tests:
        return _outcome(
            False,
            candidate_key,
            candidate,
            test_suite_sha,
            0,
            [{"test_id": "test-suite", "error_type": "no_tests", "message": "item has no tests"}],
        )
    try:
        ast.parse(candidate)
        namespace: JsonDict = {"__builtins__": _safe_builtins()}
        exec(compile(candidate, f"<{candidate_key}>", "exec"), namespace, namespace)
    except SyntaxError as exc:
        return _outcome(
            False,
            candidate_key,
            candidate,
            test_suite_sha,
            0,
            [{"test_id": "candidate-parse", "error_type": "SyntaxError", "message": exc.msg}],
        )
    except Exception as exc:  # pragma: no cover - defensive branch for malformed future items.
        return _outcome(
            False,
            candidate_key,
            candidate,
            test_suite_sha,
            0,
            [{"test_id": "candidate-load", "error_type": type(exc).__name__, "message": str(exc)}],
        )

    errors: list[JsonDict] = []
    for test in tests:
        test_id = str(test.get("test_id") or "unnamed-test")
        code = str(test.get("code") or "")
        try:
            exec(compile(code, f"<{test_id}>", "exec"), namespace, namespace)
        except AssertionError as exc:
            errors.append({"test_id": test_id, "error_type": "AssertionError", "message": str(exc)})
        except Exception as exc:
            errors.append({"test_id": test_id, "error_type": type(exc).__name__, "message": str(exc)})
    return _outcome(not errors, candidate_key, candidate, test_suite_sha, len(tests), errors)


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    config = config or ExperimentConfig()
    started = config.start_time()
    source_items = tuple(dict(item) for item in (config.manifest_items or default_items()))
    validated = _validate_items(source_items)
    accepted_rows = [row["manifest_row"] for row in validated if row["accepted"]]
    transcript_rows = [row["transcript_row"] for row in validated if row["accepted"]]

    manifest_path = config.resolved_manifest_path()
    transcript_path = config.resolved_transcript_path()
    _write_jsonl(manifest_path, accepted_rows)
    _write_jsonl(transcript_path, transcript_rows)

    all_items_have_tests = all(bool(item.get("tests")) for item in source_items)
    all_baseline_candidates_fail = all(
        row["baseline"].passed is False for row in validated if not row["flaky"]
    ) and all_items_have_tests
    all_reference_solutions_pass = all(
        row["reference"].passed is True for row in validated if not row["flaky"]
    ) and all_items_have_tests
    flaky_items = [str(row["item_id"]) for row in validated if row["flaky"]]
    ready = bool(
        20 <= len(accepted_rows) <= 40
        and all_items_have_tests
        and all_baseline_candidates_fail
        and all_reference_solutions_pass
        and not flaky_items
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    transcript_text = transcript_path.read_text(encoding="utf-8")
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT_FILENAME.removesuffix(".json"),
        "run_date": RUN_DATE,
        "hard_code_stress_set_ready": ready,
        "manifest_path": str(_relative_or_absolute(config.repo_root, manifest_path)),
        "n_items": len(accepted_rows),
        "all_items_have_tests": all_items_have_tests,
        "all_baseline_candidates_fail": all_baseline_candidates_fail,
        "all_reference_solutions_pass": all_reference_solutions_pass,
        "flaky_items": flaky_items,
        "verifier_transcript_paths": [str(_relative_or_absolute(config.repo_root, transcript_path))],
        "hard_generation_sources": _hard_generation_sources(source_items),
        "honest_verdict": (
            "ready: verifier-backed hard-code stress set validated"
            if ready
            else "blocked: hard-code stress set failed validation"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "manifest_sha256": sha256_text(manifest_text),
        "transcript_sha256": sha256_text(transcript_text),
        "validation_commands": list(VALIDATION_COMMANDS),
        "duration_s": round(config.clock() - started, 6),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "rejected_item_ids": [str(row["item_id"]) for row in validated if not row["accepted"]],
    }
    return artifact


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.resolved_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _item(
    item_id: str,
    entry_point: str,
    prompt: str,
    baseline_candidate: str,
    reference_solution: str,
    assertions: Sequence[str],
    edge_case_rationale: str,
) -> JsonDict:
    return {
        "schema_version": "carnot.repair_hard.item.v1",
        "item_id": item_id,
        "entry_point": entry_point,
        "prompt": prompt,
        "baseline_candidate": baseline_candidate,
        "reference_solution": reference_solution,
        "expected_behavior": prompt,
        "tests": [
            {"test_id": f"SCENARIO-CODE-2990-{item_id}-{index}", "code": code}
            for index, code in enumerate(assertions, start=1)
        ],
        "provenance": [
            "synthetic_edge_case_fixtures:REQ-CODE-2990",
            "research-references.md:Post-.280 HardTests/HARDTESTGEN",
            "results/experiment_2964_sota_dccd_repair_replication_v1.json:selected_repair_set",
            "results/experiment_2977_sota_intent_preserving_code_repair_v1.json:blocked_cpu_smoke",
        ],
        "edge_case_rationale": [edge_case_rationale],
        "determinism": {
            "no_random": True,
            "no_wall_clock": True,
            "no_external_io": True,
            "timeout_s": 1.0,
        },
    }


def _validate_items(items: Sequence[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in items:
        flaky = _item_has_nondeterministic_test(item)
        baseline = (
            _synthetic_flaky_outcome(item, "baseline_candidate")
            if flaky
            else run_candidate_tests(item, "baseline_candidate")
        )
        reference = (
            _synthetic_flaky_outcome(item, "reference_solution")
            if flaky
            else run_candidate_tests(item, "reference_solution")
        )
        has_tests = bool(item.get("tests"))
        accepted = has_tests and not flaky and not baseline.passed and reference.passed
        transcript = _transcript_row(item, baseline, reference, flaky)
        manifest_row = dict(item)
        manifest_row.update(
            {
                "baseline_verification": baseline.as_dict(),
                "reference_verification": reference.as_dict(),
                "transcript_sha256": sha256_text(_canonical_json(transcript)),
            }
        )
        rows.append(
            {
                "item_id": item.get("item_id"),
                "baseline": baseline,
                "reference": reference,
                "flaky": flaky,
                "accepted": accepted,
                "manifest_row": manifest_row,
                "transcript_row": transcript,
            }
        )
    return rows


def _synthetic_flaky_outcome(item: Mapping[str, Any], candidate_key: str) -> VerificationOutcome:
    tests = tuple(dict(row) for row in item.get("tests") or ())
    return _outcome(
        False,
        candidate_key,
        str(item.get(candidate_key) or ""),
        sha256_text(_canonical_json(tests)),
        0,
        [
            {
                "test_id": "determinism-scan",
                "error_type": "nondeterministic_test",
                "message": "test suite references random/time behavior",
            }
        ],
    )


def _outcome(
    passed: bool,
    candidate_key: str,
    candidate: str,
    test_suite_sha: str,
    tests_run: int,
    errors: Sequence[JsonDict],
) -> VerificationOutcome:
    failing_ids = [str(error.get("test_id") or "") for error in errors]
    return VerificationOutcome(
        passed=passed,
        candidate_key=candidate_key,
        candidate_sha256=sha256_text(candidate),
        test_suite_sha256=test_suite_sha,
        tests_run=tests_run,
        failing_test_ids=failing_ids,
        errors=[dict(error) for error in errors],
    )


def _transcript_row(
    item: Mapping[str, Any],
    baseline: VerificationOutcome,
    reference: VerificationOutcome,
    flaky: bool,
) -> JsonDict:
    return {
        "item_id": str(item.get("item_id") or ""),
        "entry_point": str(item.get("entry_point") or ""),
        "test_suite_sha256": baseline.test_suite_sha256,
        "baseline": baseline.as_dict(),
        "reference": reference.as_dict(),
        "flaky": flaky,
    }


def _item_has_nondeterministic_test(item: Mapping[str, Any]) -> bool:
    codes = " ".join(str(test.get("code") or "") for test in item.get("tests") or ())
    return any(token in codes for token in NONDETERMINISTIC_TOKENS)


def _hard_generation_sources(items: Sequence[Mapping[str, Any]]) -> list[str]:
    sources = {
        str(source)
        for item in items
        for source in item.get("provenance", ())
        if str(source).strip()
    }
    return sorted(sources)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _safe_builtins() -> JsonDict:
    return {
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _relative_or_absolute(root: Path, path: Path) -> Path:
    root_resolved = root.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    try:
        return path_resolved.relative_to(root_resolved)
    except ValueError:
        return path_resolved


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_artifact()
    return 0 if artifact["hard_code_stress_set_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
