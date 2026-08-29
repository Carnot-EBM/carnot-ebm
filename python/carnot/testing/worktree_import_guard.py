"""Refuse a test run whose `carnot` package comes from a DIFFERENT checkout than its tests.

WHY THIS EXISTS (2026-08-29). We adopted per-agent git worktrees so concurrent agents stop
destroying each other's work through the shared index and pre-commit's stash window. Worktrees
close that class. They open a quieter one.

The venv resolves `import carnot` to the MAIN checkout, wherever the process happens to be:

    $ cd <a fresh worktree> && .venv/bin/python -c "import carnot; print(carnot.__file__)"
    /home/.../carnot/python/carnot/__init__.py      <-- the MAIN checkout, not this worktree

So an agent editing `python/carnot/` inside its worktree tests the UNMODIFIED original. Every
assertion passes. Every mutation proof reads GREEN while measuring nothing. That is strictly
worse than the failure it replaced: a shared-index collision is loud and recoverable, and a
proof that silently measures the wrong tree is neither.

`PYTHONPATH=<worktree>/python` fixes it, verified both directions before this guard was written.
The guard exists because remembering to set it is exactly the kind of thing that gets forgotten
once, quietly, on the run that mattered.

FAIL-CLOSED, deliberately. A refused run is a minute of confusion with the fix in the message.
An allowed run is a green verdict about a file nobody edited.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Escape hatch for the legitimate case: testing an INSTALLED carnot against a source tree on
#: purpose. Named for what it permits, so an agent cannot set it absent-mindedly and later claim
#: it looked routine.
OVERRIDE_ENV = "CARNOT_ALLOW_FOREIGN_CARNOT_IMPORT"


def foreign_import_reason(tests_root: Path, package_dir: Path) -> str | None:
    """Explain why these two trees disagree, or None when the run is sound.

    `tests_root` is the checkout the TESTS were loaded from; `package_dir` is the directory the
    `carnot` package was actually imported from. A pure function so the rule is testable without
    a worktree fixture, following `claimed_by_other_sessions` and `determination_damage`.
    """

    tests_root = tests_root.resolve()
    package_dir = package_dir.resolve()
    # EQUALITY, NOT CONTAINMENT (corrected 2026-08-29). Containment passed two real cases it
    # should have caught, because on this machine worktrees and clones live UNDER the main root:
    #
    #   tests /home/x/carnot  +  package /home/x/carnot/.claude/worktrees/agent-abc/python/carnot
    #   tests /home/x/carnot  +  package /home/x/carnot/output/carnot-clone/python/carnot
    #
    # The first is a worktree PYTHONPATH leaking into a main-checkout run -- an agent shell's
    # export outliving its session -- which is this trap with the trees swapped. The second is
    # not hypothetical either: a scorer once swept two full repo clones nested inside the root.
    # The guard knows exactly where the package belongs, so it should say so.
    if package_dir == tests_root / "python" / "carnot":
        return None
    return (
        f"tests were loaded from {tests_root}\n"
        f"  but `import carnot` resolved to {package_dir}\n\n"
        f"  Those are different checkouts, so this run is testing code you did not edit and any\n"
        f"  result -- pass, fail, or mutation verdict -- says nothing about this worktree.\n\n"
        f"  Fix: PYTHONPATH={tests_root / 'python'} before pytest.\n"
        f"  Deliberately testing an installed carnot? Set {OVERRIDE_ENV}=1 and say so in the run."
    )


def check(tests_root: Path, package_dir: Path) -> None:
    """Raise unless the tests and the package come from the same tree."""

    if os.environ.get(OVERRIDE_ENV) == "1":
        return
    reason = foreign_import_reason(tests_root, package_dir)
    if reason is not None:
        raise RuntimeError("carnot imported from a foreign checkout:\n  " + reason)
