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
    """Check Rust test files for spec references."""
    violations = []
    for rs_file in path.rglob("*.rs"):
        content = rs_file.read_text()
        # Find all test functions
        for match in RUST_TEST_PATTERN.finditer(content):
            test_name = match.group(1)
            # Look backwards from test fn for a spec reference in comments
            start = max(0, match.start() - 500)
            context = content[start : match.end() + 500]
            if not SPEC_PATTERN.search(context):
                violations.append(f"{rs_file}::{test_name}")
    return violations


def check_python_tests(path: Path) -> list[str]:
    """Check Python test files for spec references."""
    violations = []
    for py_file in path.rglob("test_*.py"):
        content = py_file.read_text()
        for match in PYTHON_TEST_PATTERN.finditer(content):
            test_name = match.group(1)
            # Look around the test function for spec references
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
