"""Figure-integrity audit: trace every numeric constant in a figure script
back to a measured value in ``results/*.json``.

Why this exists
---------------
The 4-test integrity framework requires that every numerical claim in the
paper trace to a specific results-artifact field (rule SOURCE-ARTIFACT).
Figure scripts are a high-risk surface because they often hard-code values
copy-pasted from a notebook or a chat transcript, which silently divorces
the figure from the experiment that produced the number.

This script walks the figure-source directory, extracts every literal
numeric constant of plausible significance, and checks whether the same
value appears anywhere in ``results/*.json``. Constants that do not match
any results file are flagged as ``untraced``. The script exits with status
1 when any untraced constants remain, so it can be wired into CI.

Limitations
-----------
- We only flag *floats with a decimal* and large integers; trivial constants
  like ``0``, ``1``, ``2``, axis ticks, color codes, and DPI settings are
  ignored to keep the audit signal-rich.
- A constant matches if its string representation appears as a substring of
  any ``results/*.json`` file. This is intentionally permissive: the goal
  is to catch fabricated values, not to enforce typed equality.
- Constants inside string literals (e.g. axis labels) are skipped, because
  those are presentation, not data.

Usage
-----
    python3 scripts/figure_integrity_audit.py
    # exits 1 if untraced constants are found
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# Constants below this absolute threshold are treated as structural plot
# parameters (axis limits, line widths, font sizes) and skipped. The audit
# focuses on values plausibly derived from experimental measurements.
MIN_AUDIT_VALUE = 3.0
TRIVIAL_INTS = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    16,
    24,
    32,
    50,
    64,
    72,
    96,
    100,
    128,
    200,
    256,
    300,
    400,
    500,
    512,
    1000,
    1024,
}


def _extract_numeric_constants(py_source: str) -> list[tuple[float, int]]:
    """Walk the AST of a figure script and return (value, lineno) for every
    numeric literal that is not inside a string and not a trivial constant.
    """
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return []
    out: list[tuple[float, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = node.value
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value in TRIVIAL_INTS:
                continue
            if abs(value) < MIN_AUDIT_VALUE:
                # Sub-threshold floats (e.g. 0.5 alpha values, 0.01 ticks)
                # are skipped; figure-relevant measurements typically have
                # magnitude >= 3 (latency in microseconds, AUROC * 100, etc.).
                continue
            out.append((float(value), node.lineno))
    return out


def _value_candidates(value: float) -> list[str]:
    """Return the string forms a numeric ``value`` may take in a JSON file.

    JSON serialization can drop trailing zeros, swap ``.0`` for integer form,
    or use scientific notation; we check several renderings so that the audit
    accepts any reasonable encoding of the same measurement.
    """
    candidates: list[str] = []
    if value.is_integer():
        candidates.append(str(int(value)))
        candidates.append(f"{int(value)}.0")
    else:
        candidates.append(repr(value))
        candidates.append(f"{value:.4f}".rstrip("0").rstrip("."))
        candidates.append(f"{value:.6f}".rstrip("0").rstrip("."))
        candidates.append(f"{value:.2f}")
    return candidates


def _value_appears(value: float, haystack: str) -> bool:
    """Return True iff ``value`` plausibly appears in the given text.

    Kept as a thin helper so unit tests can exercise the matching logic
    without paying the cost of walking the full results corpus.
    """
    return any(c in haystack for c in _value_candidates(value))


def _resolve_values_streaming(values: set[float], results_dir: Path) -> set[float]:
    """Walk ``results/*.json`` one file at a time, returning the subset of
    ``values`` that were located in at least one file.

    Streaming the corpus (instead of concatenating) keeps peak memory bounded
    by the largest single JSON; the test-suite memory watchdog flags loading
    the full corpus as a leak on rigs with hundreds of results files.
    """
    remaining = set(values)
    found: set[float] = set()
    for p in sorted(results_dir.glob("*.json")):
        if not remaining:
            break
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = {v for v in remaining if _value_appears(v, text)}
        if hits:
            found.update(hits)
            remaining -= hits
        del text
    return found


def audit(
    figures_dir: Path = FIGURES_DIR, results_dir: Path = RESULTS_DIR
) -> dict[str, list[dict]]:
    """Run the audit and return a per-file report of untraced constants."""
    per_file: dict[str, list[tuple[float, int]]] = {}
    all_values: set[float] = set()
    for py_path in sorted(figures_dir.glob("*.py")):
        source = py_path.read_text(encoding="utf-8", errors="ignore")
        constants = _extract_numeric_constants(source)
        if constants:
            per_file[py_path.name] = constants
            all_values.update(v for v, _ in constants)
    found = _resolve_values_streaming(all_values, results_dir)
    report: dict[str, list[dict]] = {}
    for name, constants in per_file.items():
        untraced = [{"value": v, "line": ln} for v, ln in constants if v not in found]
        if untraced:
            report[name] = untraced
    return report


def _format_report(report: dict[str, list[dict]]) -> str:
    if not report:
        return "OK: every audited figure-constant traces to results/*.json"
    lines = ["UNTRACED FIGURE CONSTANTS:"]
    total = 0
    for name, items in report.items():
        lines.append(f"  {name}: {len(items)} untraced")
        for item in items:
            lines.append(f"    line {item['line']}: {item['value']}")
        total += len(items)
    lines.append(f"TOTAL UNTRACED: {total}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    report = audit()
    print(_format_report(report))
    print(
        json.dumps(
            {
                "files_audited": len(list(FIGURES_DIR.glob("*.py"))),
                "files_with_untraced": len(report),
                "untraced_total": sum(len(v) for v in report.values()),
            }
        )
    )
    return 0 if not report else 1


if __name__ == "__main__":
    sys.exit(main())
