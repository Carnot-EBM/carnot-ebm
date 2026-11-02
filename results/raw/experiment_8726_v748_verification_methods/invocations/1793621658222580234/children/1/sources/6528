#!/usr/bin/env python3
"""Conductor session wrapper — gates each experiment through the exclusion manifest.

**Why this exists (RETRO-067):**
    The same five experiments (308, 260, 309, 425, 410) appeared in the conductor's
    slowest-5 list for EIGHT consecutive milestones (.37 through .45), wasting
    approximately 2,870 minutes (47.8 hours) of wall-clock time.  Exp 575 built the
    exclusion manifest JSON and the ExclusionManifest class, but did not wire them
    into the conductor session — because conductor_consulted=False.

    This wrapper is the missing wire-in.  The human conductor operator (or a future
    automated conductor script) runs this wrapper BEFORE spawning any experiment agent.
    If the experiment is excluded, the wrapper exits with code 1 and prints a clear
    explanation.  If the experiment is safe to run, it exits with code 0.

**How to use in a conductor session:**

    USAGE: Before each experiment, run:
        python scripts/conductor_session_wrapper.py <exp_id>
    Exits code 1 if excluded, 0 if safe to run.

    Example:
        python scripts/conductor_session_wrapper.py 308
        # Returns exit code 1 — excluded, do not run

        python scripts/conductor_session_wrapper.py 589
        # Returns exit code 0 — safe to run

**What this does NOT do (intentional scope limit):**
    This wrapper does not modify scripts/research_conductor.py.  Wiring the check
    into the conductor itself requires human review of the conductor's task-selection
    loop.  This wrapper provides the guard function that the human (or conductor) can
    call inline.  See RETRO-067 for the full remediation plan.

Spec: REQ-INFRA-080, SCENARIO-INFRA-085, SCENARIO-INFRA-086
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path so we can import carnot.* even when
# this script is run directly from the scripts/ directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.exclusion_manifest import DEFAULT_MANIFEST_PATH, ExclusionManifest


def check_experiment(experiment_id: int, manifest_path: str | None = None) -> tuple[bool, str]:
    """Return (is_excluded, reason) for a given experiment_id.

    This function encapsulates the check_exclusion_manifest.py logic inline so
    that callers (tests, the conductor) can import and call it without subprocess
    overhead.

    Parameters
    ----------
    experiment_id : int
        The experiment ID to check (e.g. 308).
    manifest_path : str, optional
        Path to the manifest JSON.  Defaults to the standard conductor location
        (scripts/conductor_exclusion_manifest.json relative to repo root).

    Returns
    -------
    is_excluded : bool
        True if the experiment is in the exclusion manifest.
    reason : str
        Human-readable reason string if excluded, empty string if not excluded.
    """
    if manifest_path is None:
        manifest_path = str(_REPO_ROOT / DEFAULT_MANIFEST_PATH)

    manifest = ExclusionManifest(manifest_path)
    entries = manifest.load()

    if manifest.is_excluded(experiment_id):
        matching = next((e for e in entries if e.experiment_id == experiment_id), None)
        reason = matching.reason if matching else "unknown reason"
        milestone = matching.completed_milestone if matching else "unknown milestone"
        full_reason = (
            f"Experiment {experiment_id} excluded (decided at milestone {milestone}): {reason}"
        )
        return True, full_reason

    return False, ""


def print_usage() -> None:
    """Print the canonical usage instructions for human conductor operators.

    These instructions appear when the wrapper is run without an experiment_id
    argument and also in any documentation that references this script.
    """
    print(
        "USAGE: Before each experiment, run:\n"
        "    python scripts/conductor_session_wrapper.py <exp_id>\n"
        "Exits code 1 if excluded, 0 if safe to run.\n"
        "\n"
        "Examples:\n"
        "    python scripts/conductor_session_wrapper.py 308   # excluded — exits 1\n"
        "    python scripts/conductor_session_wrapper.py 589   # safe — exits 0"
    )


def main() -> None:
    """Entry point: check one experiment_id against the exclusion manifest.

    Exit codes:
        0 — experiment is NOT excluded (safe to run)
        1 — experiment IS excluded (do not run; reason printed to stderr)
        2 — bad arguments (no experiment_id or non-integer)
    """
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(2)

    try:
        exp_id = int(sys.argv[1])
    except ValueError:
        print(
            f"Error: experiment_id must be an integer, got: {sys.argv[1]}",
            file=sys.stderr,
        )
        sys.exit(2)

    is_excluded, reason = check_experiment(exp_id)

    if is_excluded:
        print(
            f"EXCLUDED: {reason}\n"
            f"  Do not run this experiment. Remove it from the roadmap or update the manifest.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Not excluded — safe to run.
    print(f"OK: Experiment {exp_id} is not in the exclusion manifest. Safe to run.")
    sys.exit(0)


if __name__ == "__main__":
    main()
