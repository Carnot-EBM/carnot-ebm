"""conductor_pre_flight.py — Print excluded experiment IDs before conductor session starts.

WHY THIS SCRIPT EXISTS:
    The conductor exclusion manifest (scripts/conductor_exclusion_manifest.json) was first
    written by Exp 666, which verified the manifest file exists and loads correctly.
    However, the conductor itself never read the manifest — conductor_consulted=null for
    15 consecutive milestones.  Experiments 380, 381, 382 (partial checkpoint) and 346
    (55M-param EORM training) appeared in the slowest-5 for THREE consecutive milestones,
    crossing the formal retirement threshold established by the Exp 308/309 precedent.

    Rather than patching the conductor directly (invasive), this pre-flight script is a
    standalone tool that the conductor can invoke as an optional first step.  It reads the
    manifest and prints each excluded experiment ID and reason to stdout.  The conductor
    (or a human) can grep the output for "Excluded experiments" to confirm the manifest
    was consulted.

    The script ALWAYS exits 0 (success), never blocking the conductor.  A missing or
    corrupt manifest is logged as a warning, not an error.

INCREMENTAL TEST SELECTION (REQ-INFRA-041, Exp 716):
    Pre-flight now uses IncrementalTestSelector to run only the tests whose coverage is
    impacted by the current git diff.  This reduces the 562-minute pre-flight overhead
    proportionally to the fraction of modules changed per cycle (typically 10-20%).

    Full-suite fallback fires automatically when:
    - The diff touches > 20 files (large-scale change), OR
    - Any crates/ file changed (Rust/PyO3 boundary requires full Python test validation).

    The selection stats (incremental_mode, tests_selected, tests_total, selection_ratio)
    are printed to stdout for the conductor log and the Exp 716 artifact.

USAGE:
    python scripts/conductor_pre_flight.py
    python scripts/conductor_pre_flight.py --manifest path/to/manifest.json
    python scripts/conductor_pre_flight.py --run-tests   # also run selected tests

Spec: REQ-INFRA-041, REQ-INFRA-095, REQ-INFRA-096,
      SCENARIO-INFRA-050, SCENARIO-INFRA-051, SCENARIO-INFRA-103, SCENARIO-INFRA-104
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Default manifest path relative to the repo root (where this script lives in scripts/).
_DEFAULT_MANIFEST = Path(__file__).parent / "conductor_exclusion_manifest.json"


def run_pre_flight(
    manifest_path: Path,
    *,
    run_tests: bool = False,
) -> int:
    """Load the exclusion manifest, print each excluded experiment, and optionally run tests.

    Always returns 0 so the conductor is never blocked by a manifest problem.
    A missing or unreadable manifest is a warning, not a fatal error — the conductor
    should still run experiments, just without exclusion filtering.

    When run_tests=True, this function also invokes the incremental test selector
    (REQ-INFRA-041) and runs pytest on the selected test files.  If the full suite
    fallback fires (diff > 20 files or Rust changed), the full suite is run instead.

    The selection stats are printed in a structured form so Exp 716 can parse them:
        [pre-flight-tests] incremental_mode=True tests_selected=12 tests_total=450 selection_ratio=0.0267

    Parameters
    ----------
    manifest_path : Path
        Absolute or relative path to the conductor_exclusion_manifest.json file.
    run_tests : bool
        When True, also run the incremental (or full) test suite via pytest.

    Returns
    -------
    int
        Always 0 (success).  Print goes to stdout so the conductor can capture it.
    """
    if not manifest_path.exists():
        print(f"[pre-flight] WARNING: manifest not found at {manifest_path} — no exclusions applied")
    else:
        try:
            raw = json.loads(manifest_path.read_text())
            entries = raw.get("excluded", [])
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[pre-flight] WARNING: manifest unreadable ({exc}) — no exclusions applied")
            entries = []

        print(f"[pre-flight] Excluded experiments ({len(entries)} total):")
        for entry in entries:
            exp_id = entry.get("experiment_id", "?")
            milestone = entry.get("completed_milestone", "unknown")
            reason = entry.get("reason", "no reason given")
            print(f"  Exp {exp_id} (milestone {milestone}): {reason}")

    if run_tests:
        _run_incremental_tests(manifest_path.parent.parent)

    return 0


def _run_incremental_tests(repo_root: Path) -> None:
    """Run the incremental or full test suite via pytest.

    Imports IncrementalTestSelector, computes the selection stats, prints them,
    and then launches pytest on the selected files (or the full suite on fallback).

    This is a separate function so it can be tested independently and so the
    exclusion-manifest logic stays clean.

    Spec: REQ-INFRA-041, SCENARIO-INFRA-050, SCENARIO-INFRA-051
    """
    try:
        # Import here to avoid making incremental_test_selector a hard dependency
        # for callers that only need the exclusion-manifest printing.
        from carnot.pipeline.incremental_test_selector import IncrementalTestSelector  # noqa: PLC0415

        selector = IncrementalTestSelector(repo_root=repo_root)
        stats = selector.get_stats()
        selected = selector.select()
    except Exception as exc:
        print(f"[pre-flight-tests] WARNING: IncrementalTestSelector failed ({exc}) — running full suite")
        stats = {"incremental_mode": False, "tests_selected": -1, "tests_total": -1, "selection_ratio": 1.0}
        selected = None

    print(
        f"[pre-flight-tests] incremental_mode={stats['incremental_mode']} "
        f"tests_selected={stats['tests_selected']} "
        f"tests_total={stats['tests_total']} "
        f"selection_ratio={stats['selection_ratio']}"
    )

    tests_dir = repo_root / "tests" / "python"
    if selected is None:
        # Full suite fallback
        print("[pre-flight-tests] Running FULL test suite (diff > 20 files or Rust changed)")
        cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-q"]
    elif len(selected) == 0:
        print("[pre-flight-tests] No Python modules changed — skipping tests")
        return
    else:
        print(f"[pre-flight-tests] Running {len(selected)} selected test files (incremental)")
        cmd = [sys.executable, "-m", "pytest"] + selected + ["-q"]

    try:
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0:
            print(f"[pre-flight-tests] WARNING: pytest exited with code {result.returncode}")
    except Exception as exc:
        print(f"[pre-flight-tests] WARNING: pytest failed to launch — {exc}")


def main() -> None:
    """Entry point: parse args, run pre-flight, exit 0."""
    parser = argparse.ArgumentParser(
        description="Print conductor exclusion manifest before session start."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to conductor_exclusion_manifest.json",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Also run incremental test suite (REQ-INFRA-041)",
    )
    args = parser.parse_args()
    sys.exit(run_pre_flight(args.manifest, run_tests=args.run_tests))


if __name__ == "__main__":
    main()
