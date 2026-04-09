#!/usr/bin/env python3
"""Dogfood Carnot's verification pipeline on its own Python source code.

Exercises the CodeExtractor, AutoExtractor, and VerifyRepairPipeline against
Carnot's own .py files to surface constraint violations, docstring/signature
mismatches, and correlate findings with test failures.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/dogfood_carnot.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure the Python package is importable from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.pipeline.extract import (
    AutoExtractor,
    CodeExtractor,
    ConstraintResult,
)


# ---------------------------------------------------------------------------
# Data structures for the report
# ---------------------------------------------------------------------------


@dataclass
class FileReport:
    """Report for a single analyzed .py file."""

    path: str
    constraints: list[ConstraintResult] = field(default_factory=list)
    violations: list[ConstraintResult] = field(default_factory=list)
    docstring_issues: list[str] = field(default_factory=list)
    parse_error: str | None = None


@dataclass
class DogfoodReport:
    """Aggregate report across all analyzed files."""

    files: list[FileReport] = field(default_factory=list)
    test_failures: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_constraints(self) -> int:
        return sum(len(f.constraints) for f in self.files)

    @property
    def total_violations(self) -> int:
        return sum(len(f.violations) for f in self.files)

    @property
    def total_docstring_issues(self) -> int:
        return sum(len(f.docstring_issues) for f in self.files)

    @property
    def files_with_violations(self) -> list[FileReport]:
        return [f for f in self.files if f.violations or f.docstring_issues]


# ---------------------------------------------------------------------------
# Step 1: Extract and verify constraints from .py files
# ---------------------------------------------------------------------------


def collect_python_files(root: Path) -> list[Path]:
    """Collect all .py files under python/carnot/, excluding __pycache__."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root / "python" / "carnot"):
        # Skip __pycache__ and hidden dirs.
        dirnames[:] = [d for d in dirnames if not d.startswith(("__pycache__", "."))]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                results.append(Path(dirpath) / fname)
    return results


def analyze_file(filepath: Path, extractor: CodeExtractor) -> FileReport:
    """Run CodeExtractor on a single .py file and collect violations."""
    rel_path = str(filepath.relative_to(REPO_ROOT))
    report = FileReport(path=rel_path)

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception as exc:
        report.parse_error = f"Could not read: {exc}"
        return report

    # CodeExtractor expects code or fenced code blocks. Feed raw source.
    try:
        constraints = extractor.extract(source, domain="code")
    except Exception as exc:
        report.parse_error = f"Extraction failed: {exc}"
        return report

    report.constraints = constraints
    report.violations = [
        c for c in constraints if c.metadata.get("satisfied") is False
    ]
    return report


# ---------------------------------------------------------------------------
# Step 2: Correlate with test failures
# ---------------------------------------------------------------------------


def run_tests_and_collect_failures() -> list[str]:
    """Run pytest in short-summary mode and return failure lines."""
    try:
        result = subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "-m",
                "pytest",
                str(REPO_ROOT / "tests" / "python"),
                "--tb=no",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return [f"Could not run tests: {exc}"]

    failures = []
    for line in result.stdout.splitlines():
        if line.startswith("FAILED"):
            failures.append(line)
    # Also capture error summary lines from stderr.
    if result.returncode != 0 and not failures:
        for line in result.stderr.splitlines()[-5:]:
            if line.strip():
                failures.append(line.strip())
    return failures


def correlate_violations_with_tests(
    file_reports: list[FileReport], test_failures: list[str]
) -> list[str]:
    """Find files that have both code violations and related test failures."""
    correlations = []
    for fr in file_reports:
        if not fr.violations:
            continue
        # Derive the module path from file path for matching.
        module_stem = fr.path.replace("/", ".").replace(".py", "")
        # Check if any test failure mentions this module or a related test.
        related_failures = []
        for fail in test_failures:
            # Match on module name fragments.
            parts = module_stem.split(".")
            if any(part in fail for part in parts if len(part) > 3):
                related_failures.append(fail)
        if related_failures:
            correlations.append(
                f"  {fr.path}: {len(fr.violations)} violation(s) "
                f"+ {len(related_failures)} related test failure(s)"
            )
    return correlations


# ---------------------------------------------------------------------------
# Step 3: Verify docstrings against function signatures
# ---------------------------------------------------------------------------


def check_docstring_args(filepath: Path) -> list[str]:
    """Check that docstring Args: sections match actual function parameters.

    Looks for functions with both type annotations and docstrings, then
    verifies that every annotated parameter appears in the Args: section
    and vice versa.
    """
    issues = []
    rel_path = str(filepath.relative_to(REPO_ROOT))

    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Skip dunder methods and private helpers — focus on public API.
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue

        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        # Extract actual parameter names (excluding 'self' and 'cls').
        actual_params = [
            arg.arg
            for arg in node.args.args
            if arg.arg not in ("self", "cls")
        ]

        # Parse the Args: section from the docstring.
        documented_params = _parse_docstring_args(docstring)

        if not documented_params and not actual_params:
            continue
        if not documented_params:
            # No Args section but has params — skip, not all need docs.
            continue

        # Check for params documented but not in signature.
        for dp in documented_params:
            if dp not in actual_params:
                issues.append(
                    f"{rel_path}:{node.lineno} {node.name}(): "
                    f"docstring documents '{dp}' but it's not in the signature"
                )

        # Check for params in signature but not documented.
        for ap in actual_params:
            if ap not in documented_params:
                issues.append(
                    f"{rel_path}:{node.lineno} {node.name}(): "
                    f"parameter '{ap}' in signature but missing from docstring Args"
                )


    return issues


def _parse_docstring_args(docstring: str) -> list[str]:
    """Extract parameter names from a Google-style Args: section."""
    params = []
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.startswith("Args:"):
            in_args = True
            continue
        if in_args:
            # End of Args section: next section header or blank line after content.
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                # Likely a new section header (Returns:, Raises:, etc.)
                if stripped.endswith(":"):
                    break
            if not stripped:
                continue
            # Parse "param_name: description" or "param_name (type): description"
            m = re.match(r"(\w+)\s*(?:\(.*?\))?\s*:", stripped)
            if m:
                params.append(m.group(1))
    return params


# ---------------------------------------------------------------------------
# Step 4: Generate report
# ---------------------------------------------------------------------------


def print_report(report: DogfoodReport) -> None:
    """Print a human-readable summary of the dogfood analysis."""
    print("=" * 72)
    print("  CARNOT DOGFOOD REPORT — Self-verification of Python source")
    print("=" * 72)
    print()
    print(f"  Files analyzed:        {len(report.files)}")
    print(f"  Constraints extracted: {report.total_constraints}")
    print(f"  Violations found:      {report.total_violations}")
    print(f"  Docstring issues:      {report.total_docstring_issues}")
    print(f"  Test failures:         {len(report.test_failures)}")
    print(f"  Elapsed:               {report.elapsed_seconds:.1f}s")
    print()

    # --- Constraint breakdown by type ---
    type_counts: dict[str, int] = {}
    type_violations: dict[str, int] = {}
    for fr in report.files:
        for c in fr.constraints:
            type_counts[c.constraint_type] = type_counts.get(c.constraint_type, 0) + 1
        for v in fr.violations:
            type_violations[v.constraint_type] = (
                type_violations.get(v.constraint_type, 0) + 1
            )

    if type_counts:
        print("  Constraint breakdown:")
        for ctype, count in sorted(type_counts.items()):
            vcount = type_violations.get(ctype, 0)
            status = f" ({vcount} violations)" if vcount else ""
            print(f"    {ctype:24s} {count:4d}{status}")
        print()

    # --- Files with violations ---
    problem_files = report.files_with_violations
    if problem_files:
        print("-" * 72)
        print("  FILES WITH ISSUES")
        print("-" * 72)
        for fr in sorted(problem_files, key=lambda f: -(len(f.violations) + len(f.docstring_issues))):
            print(f"\n  {fr.path}")
            for v in fr.violations:
                print(f"    [VIOLATION] {v.description}")
            for di in fr.docstring_issues:
                print(f"    [DOCSTRING] {di}")
    else:
        print("  No code violations or docstring issues found.")
    print()

    # --- Test failures ---
    if report.test_failures:
        print("-" * 72)
        print("  TEST FAILURES")
        print("-" * 72)
        for fail in report.test_failures[:20]:
            print(f"    {fail}")
        if len(report.test_failures) > 20:
            print(f"    ... and {len(report.test_failures) - 20} more")
        print()

    # --- Correlations ---
    correlations = correlate_violations_with_tests(report.files, report.test_failures)
    if correlations:
        print("-" * 72)
        print("  VIOLATION ↔ TEST FAILURE CORRELATIONS")
        print("-" * 72)
        for corr in correlations:
            print(corr)
        print()

    # --- Recommendations ---
    print("-" * 72)
    print("  RECOMMENDATIONS")
    print("-" * 72)
    if report.total_violations > 0:
        init_violations = sum(
            1
            for fr in report.files
            for v in fr.violations
            if v.constraint_type == "initialization"
        )
        if init_violations:
            print(
                f"    • {init_violations} initialization violation(s): variables used "
                "before assignment. These may be false positives from module-level "
                "imports or closures — review each case."
            )
        type_violations_count = sum(
            1
            for fr in report.files
            for v in fr.violations
            if v.constraint_type == "return_value_type"
        )
        if type_violations_count:
            print(
                f"    • {type_violations_count} return type mismatch(es): function "
                "returns a value whose type doesn't match the annotation."
            )
    if report.total_docstring_issues > 0:
        print(
            f"    • {report.total_docstring_issues} docstring/signature mismatch(es): "
            "update docstrings to match current function signatures."
        )
    if report.total_violations == 0 and report.total_docstring_issues == 0:
        print("    • No issues found. Carnot's own code passes self-verification.")
    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full dogfood analysis pipeline."""
    t0 = time.monotonic()

    print("Collecting Python source files...")
    py_files = collect_python_files(REPO_ROOT)
    print(f"  Found {len(py_files)} files under python/carnot/")

    # Step 1: Extract constraints from each file.
    print("Extracting code constraints...")
    extractor = CodeExtractor()
    file_reports: list[FileReport] = []
    for filepath in py_files:
        fr = analyze_file(filepath, extractor)
        file_reports.append(fr)

    # Step 3: Check docstrings.
    print("Checking docstrings against signatures...")
    for fr in file_reports:
        filepath = REPO_ROOT / fr.path
        fr.docstring_issues = check_docstring_args(filepath)

    # Step 2: Run tests and collect failures.
    print("Running test suite (this may take a moment)...")
    test_failures = run_tests_and_collect_failures()

    elapsed = time.monotonic() - t0

    report = DogfoodReport(
        files=file_reports,
        test_failures=test_failures,
        elapsed_seconds=elapsed,
    )

    print_report(report)

    # Exit with non-zero if violations found.
    if report.total_violations > 0 or report.total_docstring_issues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
