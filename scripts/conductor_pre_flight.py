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

USAGE:
    python scripts/conductor_pre_flight.py
    python scripts/conductor_pre_flight.py --manifest path/to/manifest.json

Spec: REQ-INFRA-095, REQ-INFRA-096, SCENARIO-INFRA-103, SCENARIO-INFRA-104
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Default manifest path relative to the repo root (where this script lives in scripts/).
_DEFAULT_MANIFEST = Path(__file__).parent / "conductor_exclusion_manifest.json"


def run_pre_flight(manifest_path: Path) -> int:
    """Load the exclusion manifest and print each excluded experiment.

    Always returns 0 so the conductor is never blocked by a manifest problem.
    A missing or unreadable manifest is a warning, not a fatal error — the conductor
    should still run experiments, just without exclusion filtering.

    Parameters
    ----------
    manifest_path : Path
        Absolute or relative path to the conductor_exclusion_manifest.json file.

    Returns
    -------
    int
        Always 0 (success).  Print goes to stdout so the conductor can capture it.
    """
    if not manifest_path.exists():
        print(f"[pre-flight] WARNING: manifest not found at {manifest_path} — no exclusions applied")
        return 0

    try:
        raw = json.loads(manifest_path.read_text())
        entries = raw.get("excluded", [])
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[pre-flight] WARNING: manifest unreadable ({exc}) — no exclusions applied")
        return 0

    print(f"[pre-flight] Excluded experiments ({len(entries)} total):")
    for entry in entries:
        exp_id = entry.get("experiment_id", "?")
        milestone = entry.get("completed_milestone", "unknown")
        reason = entry.get("reason", "no reason given")
        print(f"  Exp {exp_id} (milestone {milestone}): {reason}")

    return 0


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
    args = parser.parse_args()
    sys.exit(run_pre_flight(args.manifest))


if __name__ == "__main__":
    main()
