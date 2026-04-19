#!/usr/bin/env python3
"""Pre-commit hook that keeps ``docs/technical-report.html`` in sync with
``docs/technical-report.md``.

**Why this exists:** humans and the conductor's doc-reconciliation agent
both edit the Markdown version of the technical report. Neither of them
updates the HTML rendering. Without this hook the HTML drifts behind the
Markdown, which is exactly what happened before milestone 2026.04.37 when
the published GitHub Pages page was three milestones stale.

**What it does:** when ``docs/technical-report.md`` is part of the commit
(``pre-commit`` already filters to that path via the ``files`` regex in
``.pre-commit-config.yaml``), run the build script to regenerate the HTML
and stage the result so it rides along in the same commit as the Markdown
change. This turns "one file edited, the other is stale" into a bounded
never-state: the commit either contains both or neither.

**Why not use ``build_technical_report.py --check`` directly:** ``--check``
fails the commit and leaves the user to re-run the build and re-stage.
That works, but the conductor's reconciliation Haiku is a non-interactive
agent -- it will not "re-run and re-stage" on its own, so check-mode alone
would just break the conductor's commit pipeline. This wrapper does the
regen and the ``git add`` itself, so the same hook works for both human
commits and the conductor's automated commits.

**Exit codes:** 0 on success (either regenerated and staged, or nothing to
do). Non-zero only if the build script itself errors out, in which case
pre-commit surfaces the traceback.

Run directly (not as a hook) for local diagnostics:
    python scripts/precommit_sync_technical_report.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_REL_PATH = "docs/technical-report.md"
HTML_REL_PATH = "docs/technical-report.html"


def _run(cmd: list[str]) -> str:
    """Run ``cmd`` from the project root, return stdout stripped.

    Raises ``subprocess.CalledProcessError`` on non-zero exit so pre-commit
    surfaces the failure rather than silently pretending everything is fine.
    """
    return subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def md_is_staged() -> bool:
    """Return True if docs/technical-report.md is in the staged changeset.

    pre-commit already applies a ``files`` regex filter in
    ``.pre-commit-config.yaml``, so in practice this function is redundant
    for the hook entry. We keep it so the script is safe to run standalone
    (e.g. by a developer checking what the hook would do) without always
    forcing a regen.
    """
    staged = _run(["git", "diff", "--cached", "--name-only"]).splitlines()
    return MD_REL_PATH in staged


def main() -> int:
    if not md_is_staged():
        # The pre-commit framework's ``files`` filter usually prevents this
        # branch from being reached when run as a hook, but if invoked by
        # hand without staged changes there is nothing to do.
        return 0

    build_script = PROJECT_ROOT / "scripts" / "build_technical_report.py"
    # Use the same Python that is running this hook. The project's venv
    # has python-markdown installed; the system Python may not.
    _run([sys.executable, str(build_script)])

    # Stage the regenerated HTML. If it was already up to date this is a
    # no-op as far as the resulting commit is concerned (the file's content
    # did not change, so ``git diff --cached`` will not show it).
    _run(["git", "add", HTML_REL_PATH])
    return 0


if __name__ == "__main__":
    sys.exit(main())
