#!/usr/bin/env python3
"""Audit metric provenance across experiment deliverables.

Walks `results/experiment_*.json`, extracts the `metrics_provenance` field
(if present), and reports:

  1. Deliverables WITH provenance, grouped by (function, version) tuple.
  2. Deliverables WITHOUT provenance (pre-2026-04-28 deliverables; the
     full set unless retroactively backfilled).
  3. Deliverables flagged for retrospective re-evaluation when a metric
     implementation is bumped to a new version.

Usage:
    # Show full audit
    python3 scripts/audit_metric_provenance.py

    # Flag deliverables using a known-buggy version
    python3 scripts/audit_metric_provenance.py --flag-buggy auroc:v0.x

This is the operational tool that makes "we found a bug — what does it
taint?" a 30-second query rather than a 30-minute manual grep+interpret
pass. Origin: 2026-04-28 inverted-AUROC discovery in `exp995` and
`exp1003` required exactly that manual audit. Spec: REQ-EVAL-004.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flag-buggy",
        nargs="*",
        default=[],
        metavar="FUNC:VERSION",
        help=(
            "Mark `(function, version)` pairs as known-buggy. Deliverables "
            "tagged with these are listed for retrospective re-evaluation. "
            "Example: --flag-buggy auroc:v0.x"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(_RESULTS_DIR),
        help=f"Override results directory (default {_RESULTS_DIR})",
    )
    args = parser.parse_args()

    flagged_pairs = set()
    for spec in args.flag_buggy:
        if ":" not in spec:
            print(
                f"warning: --flag-buggy {spec!r} not in func:version format, skipping",
                file=sys.stderr,
            )
            continue
        flagged_pairs.add(spec.strip())

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 2

    by_provenance: dict[str, list[str]] = defaultdict(list)
    no_provenance: list[str] = []
    flagged_deliverables: list[tuple[str, str]] = []

    for path in sorted(results_dir.glob("experiment_*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            no_provenance.append(path.name)
            continue
        prov = data.get("metrics_provenance")
        if not isinstance(prov, dict) or not prov:
            no_provenance.append(path.name)
            continue
        for func_name, version_string in prov.items():
            by_provenance[version_string].append(path.name)
            # Match either exact `func:version` or just `func:` prefix
            for flag in flagged_pairs:
                flag_func = flag.split(":", 1)[0]
                if version_string.startswith(f"carnot.eval.metrics.{flag_func}:") and (
                    flag in version_string
                    or flag.endswith(":")
                    or flag.split(":", 1)[1] in version_string
                ):
                    flagged_deliverables.append((path.name, version_string))

    print("=" * 72)
    print("Metrics Provenance Audit")
    print("=" * 72)
    print()
    print(
        f"Total experiment_*.json files: {len(no_provenance) + sum(len(v) for v in by_provenance.values())}"
    )
    print(f"  With metrics_provenance: {sum(len(v) for v in by_provenance.values())}")
    print(f"  Without (pre-2026-04-28 or no metrics): {len(no_provenance)}")
    print()

    if by_provenance:
        print("By (function, version):")
        for version_string, deliverables in sorted(by_provenance.items()):
            print(f"  {version_string}: {len(deliverables)} deliverable(s)")
            if len(deliverables) <= 5:
                for d in deliverables:
                    print(f"    - {d}")
        print()

    if flagged_deliverables:
        print("=" * 72)
        print(f"FLAGGED FOR RE-EVALUATION ({len(flagged_deliverables)} deliverable(s))")
        print("=" * 72)
        for deliverable, version in flagged_deliverables:
            print(f"  {deliverable} → {version}")
        print()
        return 1

    if not flagged_pairs and no_provenance:
        print("=" * 72)
        print(f"PRE-PROVENANCE BACKLOG ({len(no_provenance)} deliverable(s) without tag)")
        print("=" * 72)
        print(
            "These were produced before metrics_provenance plumbing landed. "
            "If you need to retroactively audit, grep for AUROC/F1 values "
            "and cross-reference with `git log scripts/experiment_*.py`."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
