#!/usr/bin/env python3
"""Pre-commit hook entry point for batching enforcement.

**Why this hook exists (RETRO-045):**
    Exp 481 documented 77 sequential-loop violations in experiment scripts and wrote
    the BatchedInferenceRunner standard.  Documentation alone is insufficient: the
    conductor continued writing new scripts with sequential for-loops because nothing
    prevented it at commit time.

    This hook closes RETRO-045 by making enforcement automatic.  When a new script
    with a sequential question loop (``for q in questions:``) is staged for commit,
    the hook exits 1, preventing the commit until the loop is migrated to
    ``BatchedInferenceRunner``.  This converts a voluntary standard into a hard gate.

**How it works:**
    1. Reads staged files from ``git diff --cached --name-only --diff-filter=ACM``.
    2. Filters to ``scripts/*.py`` files only (other Python files are not experiment
       scripts and do not need the batching standard).
    3. Runs ``BatchingHookRunner(scripts_dir, staged_scripts).run()``.
    4. If any high-severity violations are found, prints them and exits 1.
    5. If no violations, exits 0 (commit proceeds).

**Idempotency (REQ-INFRA-053):**
    Only violations in the STAGED files are reported.  Pre-existing violations in
    already-committed scripts are ignored, so the hook does not block unrelated commits
    until all 77 historical violations are fixed.

Spec: REQ-INFRA-052, REQ-INFRA-053,
      SCENARIO-INFRA-060, SCENARIO-INFRA-061
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

# Ensure the project Python package is importable when run as a hook.
# pre-commit runs hooks in a subprocess where sys.path may not include the project root.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "python"))

from carnot.pipeline.batching_hook_runner import BatchingHookRunner  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


def get_staged_files() -> list[str]:
    """Return paths of staged Python files from git diff --cached.

    Uses ``--diff-filter=ACM`` to include only Added, Copied, and Modified files.
    Deleted files are excluded because they cannot have new violations.

    Returns
    -------
    list[str]
        Relative paths of staged files from repo root.  Empty list when git is not
        available or no Python files are staged.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not inside a git repo or git not installed — skip the check.
        return []


def main() -> int:
    """Run batching enforcement check on staged scripts/ files.

    Returns
    -------
    int
        0 if no high-severity violations found, 1 otherwise.
    """
    all_staged = get_staged_files()

    # Filter to scripts/*.py only — experiment scripts are the target of the standard.
    # Library code in python/carnot/ and test code are excluded.
    staged_scripts = [
        f for f in all_staged
        if f.startswith("scripts/") and f.endswith(".py")
    ]

    if not staged_scripts:
        # No experiment scripts staged — nothing to check.
        return 0

    scripts_dir = str(_repo_root / "scripts")
    runner = BatchingHookRunner(scripts_dir=scripts_dir, staged_files=[
        str(_repo_root / sf) for sf in staged_scripts
    ])

    violations = runner.run(raise_on_violation=True)

    if violations:
        print("\n[batching-check] FAILED: sequential question loops detected in staged scripts.")
        print("[batching-check] Migrate to BatchedInferenceRunner before committing.")
        print("[batching-check] See CLAUDE.md Experiment Template section and Exp 437/481.")
        print()
        for v in violations:
            print(f"  {v.script_path}:{v.line_no}: [{v.severity}] {v.pattern}")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
