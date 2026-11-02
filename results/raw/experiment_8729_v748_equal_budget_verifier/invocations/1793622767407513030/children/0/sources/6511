#!/usr/bin/env python3
"""Check whether a given experiment_id is excluded from conductor re-entry.

Usage::

    python scripts/check_exclusion_manifest.py 308
    # exits with code 1 and prints an error message if 308 is excluded
    # exits with code 0 if 308 is not excluded (safe to run)

This script is intended for use in the conductor session-start checklist.
Run it before spawning an agent for any experiment to avoid re-running
experiments that are stuck or superseded (RETRO-056).

Spec: REQ-INFRA-071, SCENARIO-INFRA-076
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.exclusion_manifest import DEFAULT_MANIFEST_PATH, ExclusionManifest


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: check_exclusion_manifest.py <experiment_id>", file=sys.stderr)
        sys.exit(2)

    try:
        exp_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: experiment_id must be an integer, got: {sys.argv[1]}", file=sys.stderr)
        sys.exit(2)

    manifest_path = _REPO_ROOT / DEFAULT_MANIFEST_PATH
    manifest = ExclusionManifest(str(manifest_path))
    entries = manifest.load()

    if manifest.is_excluded(exp_id):
        # Find the matching entry for a helpful error message.
        matching = next((e for e in entries if e.experiment_id == exp_id), None)
        reason = matching.reason if matching else "unknown reason"
        milestone = matching.completed_milestone if matching else "unknown milestone"
        print(
            f"EXCLUDED: Experiment {exp_id} is in the conductor exclusion manifest.\n"
            f"  Decided at milestone: {milestone}\n"
            f"  Reason: {reason}\n"
            f"  Do not run this experiment. Remove it from the roadmap or update the manifest.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Not excluded — exit 0 (success).


if __name__ == "__main__":
    main()
