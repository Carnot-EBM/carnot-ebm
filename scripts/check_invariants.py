#!/usr/bin/env python3
"""Check machine-readable invariants against experiment result artifacts.

Run after a milestone (or any time) to catch honest-verdict strings whose
data contradicts them.  Exits 0 when every artifact passes; exits 1 when
any artifact fails at least one invariant.  The report is printed to stdout
as human-readable text PLUS written to a JSON file for programmatic use.

Typical usage — audit a milestone at retro time:

    .venv/bin/python scripts/check_invariants.py results/experiment_678*.json \\
        results/experiment_679*.json results/experiment_680*.json \\
        results/experiment_681*.json ...

    # or, audit the whole repo:
    .venv/bin/python scripts/check_invariants.py "results/experiment_*.json"

Would have caught (and will catch on the next similar failure):

    - Exp 652, 669, original-678: distillation verdicts with duration < 30s
    - Exp 679: vr_positive with baseline_accuracy=0.0
    - Exp 691: verified_publishable with TP=0 on every dataset
    - Exp 691: OOD AUROC 0.16 higher than in-distribution AUROC

See ``python/carnot/invariants.py`` for the full invariant list and the
conditions under which each fires.  Add new invariants there, not here.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from carnot.invariants import run_invariants  # noqa: E402


def _collect_result_files(patterns: list[str]) -> list[Path]:
    """Expand shell-style globs into concrete result-file paths.

    Accepts either concrete paths or glob patterns.  Returns deduplicated,
    sorted, existing-files-only paths.
    """
    hits: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file() and p.suffix == ".json":
                hits.add(p.resolve())
    return sorted(hits)


def _load_artifact(path: Path) -> dict[str, Any] | None:
    """Load one artifact.  Return None on unparseable JSON — a bad file should
    not crash the whole audit; it becomes its own report entry."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _format_violation_text(path: Path, violations: list) -> str:
    lines = [f"\n  {path.relative_to(_REPO_ROOT) if path.is_relative_to(_REPO_ROOT) else path}:"]
    for v in violations:
        lines.append(f"    ✗ {v.invariant_name}")
        if v.reason:
            lines.append(f"        reason: {v.reason}")
        if v.suggested_verdict:
            lines.append(f"        suggested_verdict: {v.suggested_verdict}")
        if v.evidence:
            lines.append(f"        evidence: {v.evidence}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check invariants on one or more experiment result JSONs.",
    )
    parser.add_argument(
        "patterns", nargs="+",
        help="Paths or glob patterns for result JSONs.",
    )
    parser.add_argument(
        "--report", default=None,
        help="Optional path to write the JSON report.  Defaults to "
             "results/invariant_check_report.json when violations are found.",
    )
    args = parser.parse_args()

    files = _collect_result_files(args.patterns)
    if not files:
        print("check_invariants: no result JSONs matched the given patterns.", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "n_files_checked": len(files),
        "n_files_with_violations": 0,
        "total_violations": 0,
        "per_file": [],
    }

    print(f"check_invariants: auditing {len(files)} file(s)...")
    for path in files:
        artifact = _load_artifact(path)
        if artifact is None:
            print(f"  ? {path}: could not parse as JSON (skipped)")
            report["per_file"].append({
                "path": str(path),
                "parseable": False,
                "violations": [],
            })
            continue
        violations = run_invariants(artifact)
        entry: dict[str, Any] = {
            "path": str(path),
            "parseable": True,
            "honest_verdict": artifact.get("honest_verdict"),
            "experiment": artifact.get("experiment"),
            "violations": [v.as_dict() for v in violations],
        }
        report["per_file"].append(entry)
        if violations:
            report["n_files_with_violations"] += 1
            report["total_violations"] += len(violations)
            print(_format_violation_text(path, violations))

    print()
    print(
        f"Summary: {report['n_files_with_violations']} / {len(files)} files "
        f"have invariant violations ({report['total_violations']} total)."
    )

    if report["total_violations"] > 0 or args.report is not None:
        out_path = Path(args.report) if args.report else (
            _REPO_ROOT / "results" / "invariant_check_report.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Report written to {out_path.relative_to(_REPO_ROOT) if out_path.is_relative_to(_REPO_ROOT) else out_path}")

    return 1 if report["total_violations"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
