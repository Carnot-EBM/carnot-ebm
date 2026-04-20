#!/usr/bin/env python3
"""Conductor manifest pre-check — proves conductor_consulted=True before any experiment runs.

**Why this exists (RETRO-067):**
    The same five experiments (308, 260, 309, 425, 410) appeared in the conductor's
    slowest-5 list for NINE consecutive milestones (.37 through .45), wasting
    approximately 3,255 minutes (54.3 hours).  The exclusion manifest was built
    (Exp 575) and a wire-in wrapper was created (Exp 589), but both required the human
    conductor to explicitly call them.  The .45 retrospective confirmed
    conductor_consulted=False because neither was called automatically.

    This pre-check script PROVES consultation happened by writing a sentinel file
    (scripts/conductor_consulted_at.txt) with the current timestamp.  Any downstream
    tool can read that file's mtime to verify the conductor checked the manifest
    within the last 60 seconds before spawning an experiment agent.

**How to use in a conductor session (MANDATORY):**

    Before running ANY experiment, call:
        python scripts/conductor_manifest_precheck.py <exp_id> [<exp_id2> ...]

    Exit codes:
        0 — all experiments are safe to run; sentinel written
        1 — one or more experiments are excluded; do NOT run them
        2 — bad arguments (no experiment IDs)

    Example:
        python scripts/conductor_manifest_precheck.py 308
        # Prints [EXCLUDED] and exits 1 — do NOT run

        python scripts/conductor_manifest_precheck.py 601
        # Prints [PRECHECK OK] conductor_consulted=True and exits 0 — safe to run

**How the sentinel proves consultation:**
    The file scripts/conductor_consulted_at.txt is written (or overwritten) each time
    this script runs successfully (exit 0).  The conductor or any audit tool can check:
        mtime of conductor_consulted_at.txt < 60 seconds ago
    to verify that a human/conductor consulted the manifest before spawning the agent.

Spec: REQ-INFRA-085, SCENARIO-INFRA-090, SCENARIO-INFRA-091
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path so carnot.* is importable when this
# script is run directly from the scripts/ directory without pip install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.exclusion_manifest import DEFAULT_MANIFEST_PATH, ExclusionManifest  # noqa: E402

_MANIFEST_PATH = str(_REPO_ROOT / DEFAULT_MANIFEST_PATH)
_SENTINEL_PATH = _REPO_ROOT / "scripts" / "conductor_consulted_at.txt"


def _utc_now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_precheck(exp_ids: list[int], manifest_path: str | None = None) -> tuple[bool, list[int]]:
    """Check a list of experiment IDs against the exclusion manifest.

    This is the importable form of the pre-check logic — tests and the conductor
    can call this without subprocess overhead.

    Parameters
    ----------
    exp_ids : list[int]
        The experiment IDs to check (e.g. [308, 601]).
    manifest_path : str, optional
        Path to the manifest JSON.  Defaults to the standard conductor location.

    Returns
    -------
    all_safe : bool
        True if NO experiment in exp_ids is excluded.
    excluded_ids : list[int]
        The subset of exp_ids that ARE excluded (empty if all_safe is True).
    """
    path = manifest_path or _MANIFEST_PATH
    manifest = ExclusionManifest(path)
    entries = manifest.load()

    # Build a lookup from id -> entry for error messages.
    entry_by_id = {e.experiment_id: e for e in entries}

    excluded_found: list[int] = []
    for exp_id in exp_ids:
        if manifest.is_excluded(exp_id):
            entry = entry_by_id.get(exp_id)
            reason = entry.reason if entry else "unknown reason"
            milestone = entry.completed_milestone if entry else "unknown"
            print(
                f"[EXCLUDED] Exp {exp_id} is in exclusion manifest — skipping. "
                f"(Decided at milestone {milestone}: {reason})"
            )
            excluded_found.append(exp_id)

    return len(excluded_found) == 0, excluded_found


def write_sentinel() -> str:
    """Write the conductor_consulted_at.txt sentinel file and return the timestamp.

    The sentinel's mtime proves that the conductor ran the pre-check recently.
    Any tool that requires proof-of-consultation can check:
        time.time() - os.path.getmtime(sentinel_path) < 60
    """
    ts = _utc_now()
    _SENTINEL_PATH.write_text(f"{ts}\n")
    return ts


def main() -> None:
    """Entry point: check one or more experiment IDs and write the sentinel.

    Exit codes:
        0 — all IDs safe; sentinel written; prints [PRECHECK OK] conductor_consulted=True
        1 — one or more IDs excluded; sentinel NOT written; [EXCLUDED] lines printed
        2 — bad arguments
    """
    if len(sys.argv) < 2:
        print(
            "USAGE: python scripts/conductor_manifest_precheck.py <exp_id> [<exp_id2> ...]\n"
            "Exits 1 if any experiment is excluded, 0 if all are safe.",
            file=sys.stderr,
        )
        sys.exit(2)

    exp_ids: list[int] = []
    for arg in sys.argv[1:]:
        try:
            exp_ids.append(int(arg))
        except ValueError:
            print(f"Error: experiment_id must be an integer, got: {arg}", file=sys.stderr)
            sys.exit(2)

    all_safe, excluded = run_precheck(exp_ids)

    if not all_safe:
        sys.exit(1)

    # All safe — write the sentinel and confirm.
    ts = write_sentinel()
    print(f"[PRECHECK OK] conductor_consulted=True (sentinel written at {ts})")
    sys.exit(0)


if __name__ == "__main__":
    main()
