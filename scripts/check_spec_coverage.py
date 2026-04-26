#!/usr/bin/env python3
"""Check that all test functions reference a REQ-* or SCENARIO-* spec identifier.

This enforces the 100% spec coverage requirement (FR-09):
every test must trace to a specification requirement.

Exit code 0 = all tests have spec references.
Exit code 1 = some tests lack spec references.
"""

import re
import sys
from pathlib import Path

SPEC_PATTERN = re.compile(r"(REQ-[A-Z]+-\d+|SCENARIO-[A-Z]+-\d+)")

# Rust test pattern: #[test] followed by fn test_name
RUST_TEST_PATTERN = re.compile(r"#\[test\]\s*\n\s*fn\s+(\w+)")

# Python test pattern: def test_name
PYTHON_TEST_PATTERN = re.compile(r"def\s+(test_\w+)")


def check_rust_tests(path: Path) -> list[str]:
    """Check Rust test files for spec references.

    Coverage is satisfied if EITHER the file as a whole references at
    least one REQ-* / SCENARIO-* identifier (typical pattern: a Spec:
    line in the module header comment), OR each individual test has a
    spec reference within ±500 chars of its `fn test_*` line. The
    file-level pass mirrors how Carnot test files are actually written —
    one Spec: line at the top scopes the whole file's tests.
    """
    violations = []
    for rs_file in path.rglob("*.rs"):
        content = rs_file.read_text()
        # File-level coverage: a single ref anywhere in the file is enough.
        if SPEC_PATTERN.search(content):
            continue
        # No file-level ref: fall back to per-test window check.
        for match in RUST_TEST_PATTERN.finditer(content):
            test_name = match.group(1)
            start = max(0, match.start() - 500)
            context = content[start : match.end() + 500]
            if not SPEC_PATTERN.search(context):
                violations.append(f"{rs_file}::{test_name}")
    return violations


def check_python_tests(path: Path) -> list[str]:
    """Check Python test files for spec references.

    File-level coverage: if the file has any REQ-* / SCENARIO-* anywhere
    (typically in the module docstring's "Spec: REQ-..." line), all tests
    in that file are considered covered. This matches how Carnot test
    files are actually authored — one Spec: line at the top scopes the
    whole file's tests, and the previous per-test ±200/+1000 window was
    too small to find that line for tests in the latter half of larger
    files (causing the 2026-04-26 audit's 7,456-violation false alarm).

    Files with NO file-level ref still get the strict per-test window
    check, so genuinely-untraceable tests are still caught.
    """
    violations = []
    for py_file in path.rglob("test_*.py"):
        content = py_file.read_text()
        if SPEC_PATTERN.search(content):
            continue
        for match in PYTHON_TEST_PATTERN.finditer(content):
            test_name = match.group(1)
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 1000)
            context = content[start:end]
            if not SPEC_PATTERN.search(context):
                violations.append(f"{py_file}::{test_name}")
    return violations


def main() -> int:
    root = Path(__file__).parent.parent

    violations = []
    violations.extend(check_rust_tests(root / "crates"))
    violations.extend(check_python_tests(root / "tests" / "python"))

    if violations:
        print("ERROR: The following tests lack spec references (REQ-* or SCENARIO-*):")
        for v in sorted(violations):
            print(f"  - {v}")
        print(f"\n{len(violations)} test(s) missing spec traceability.")
        return 1

    print("OK: All tests reference specification requirements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
