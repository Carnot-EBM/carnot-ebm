#!/usr/bin/env python3
"""Sealed hashes and scope globs: stop an agent widening its own blast radius.

WHY THIS EXISTS (2026-08-27)
----------------------------
Two failures on the same day, both from the same root: nothing in this repo records what an
agent said it was going to touch, so nothing can tell an intended edit from a stray one.

  1. `git add -A` swept an unrelated staged file into another agent's commit, twice. Content
     survived both times, but one agent then searched the object store, concluded a
     colleague's work was destroyed, and offered to rewrite it. Two agents independently
     invented "commit with an explicit pathspec" as the fix, under fire.
  2. A mutation marker armed by one agent's pytest named eight files, listed all eight as
     `unattributed`, and refused every commit in the repo until a human read it. The detector
     diffs the whole tree, so it cannot say whose write a change was.

The gap both share: a check that looks at the tree can only ask "did this change?", never
"was this change yours to make?". That second question needs a declaration, made BEFORE the
work, of what the agent intends to touch.

WHAT THIS ENFORCES
------------------
An agent declares a scope up front. From then until it releases the scope:

  * SCOPE GLOBS. Every staged path must match one of the declared globs. Anything else
    refuses the commit. This is the mechanical form of "commit with a pathspec" -- but it
    cannot be forgotten under pressure, which is exactly when it was forgotten.

  * SEALED HASHES. A set of harness files -- the guards, the gate config, the test
    infrastructure -- is hashed at declaration time and re-hashed at commit time. A drift
    refuses the commit unless that exact path was named in the declaration.

    Sealed files are checked in the WORKING TREE, not the index, and that is the whole point.
    An agent that weakens a test file to make its own change pass does not need to stage it:
    the suite reads the working tree. A staged-diff check cannot see that at all.

TWO PROPERTIES THAT MAKE IT WORTH HAVING
----------------------------------------
DECLARED BEFORE, NOT AFTER. The scope is written before the work, so it cannot be widened
afterwards to cover something that turned out to be inconvenient. A scope file may not be
staged by the commit it is governing -- see `_scope_self_staged`. Without that rule the whole
control is decorative: an agent that can edit its own scope in the same commit has no scope.

SEALED PATHS NEED AN EXPLICIT NAME, NEVER A GLOB. `--scope 'scripts/**'` does NOT unseal
`scripts/adversarial_verify.py`. Widening a glob is cheap and is the thing an agent under
pressure does; typing the name of the fabrication gate is a deliberate act, and it leaves a
record of the intent in the declaration.

FAIL DIRECTION, STATED OUT LOUD
-------------------------------
With NO declaration this check is INERT and exits 0. That is deliberate, and it is the
difference between a guard that ships and one that wedges the research loop on its first day:
the conductor commits with `git add -A` every few minutes and declares nothing, so it is
unaffected. Inside a declaration the check is FAIL-CLOSED -- an unreadable scope file refuses
rather than passes, because unreadable is not the same as absent.

The honest limit of that choice: this cannot stop an agent that never declares a scope. It is
a seatbelt, not a cage. What it does buy is that an agent which HAS declared one cannot
quietly grow it, and that is the failure that actually happened here.

USAGE
-----
    # Before the work. Prints the run id and the exact --release line.
    python3 scripts/harness_integrity_lint.py --declare \\
        --scope 'scripts/foo.py' --scope 'tests/python/test_foo.py'

    # To also permit editing a sealed harness file, name it. A glob will not do it.
    python3 scripts/harness_integrity_lint.py --declare \\
        --scope 'scripts/foo.py' --unseal .pre-commit-config.yaml

    # The pre-commit gate. No arguments.
    python3 scripts/harness_integrity_lint.py --check

    # After the work.
    python3 scripts/harness_integrity_lint.py --release

Exit 0 = nothing declared, or every staged path is in scope and every seal holds.
Exit 1 = a staged path is out of scope, or a sealed file moved. Both are named.

Spec: REQ-INFRA-6800 (scope globs), REQ-INFRA-6801 (sealed hashes),
SCENARIO-INFRA-6802..6807.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# Derived from __file__, never a hardcoded absolute path: CLAUDE.md "Test-Run Record
# Integrity" rule 4 -- a baked-in root makes a fresh clone write into the operator's checkout.
REPO = Path(__file__).resolve().parents[1]
SCOPES = REPO / "ops" / ".agent_scopes"

#: Pin one id for a whole workflow so --check is unambiguous when sessions share a shell.
RUN_ID_ENV = "CARNOT_AGENT_SCOPE_ID"

# The harness: files an agent could edit to make its own work pass. This is not a list of
# important files -- it is a list of files whose modification changes the VERDICT on other
# files. That is why the fabrication gate is here and, say, the README is not (the README has
# its own guard, for a different reason: see scripts/operator_curated_docs_lint.py).
#
# Keep it SHORT. Every entry is a path an agent must name explicitly to touch, so a long list
# is a tax on honest work, and a tax on honest work is how a guard teaches people to bypass it.
SEALED_PATHS: tuple[str, ...] = (
    ".pre-commit-config.yaml",
    "scripts/adversarial_verify.py",
    "scripts/determination_preservation_lint.py",
    "scripts/test_suite_mutation_check.py",
    "scripts/qa_layer_authenticity_audit.py",
    "scripts/operator_curated_docs_lint.py",
    "scripts/harness_integrity_lint.py",
    "python/carnot/testing/operator_curated_doc_guard.py",
    "tests/python/conftest.py",
)


def _git(*args: str) -> str:
    """Run git and return stdout, or "" if git itself failed.

    Callers must treat "" as UNKNOWN and refuse, never as "nothing found" -- a guard that
    reports clean when git is broken is the trusted-and-silent state CLAUDE.md warns about.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def _sha256(path: Path) -> str | None:
    """Hash a file, or None when it does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _auto_run_id() -> str:
    """An id containing the parent PID and a timestamp.

    The PID alone does not discriminate here: agent workflows are frequently children of the
    same shell, so two declarations would collide. The timestamp is what separates them.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"ppid-{os.getppid()}-{stamp}"


def _resolve_run_id(explicit: str | None) -> str:
    return explicit or os.environ.get(RUN_ID_ENV) or _auto_run_id()


def _scope_files() -> list[Path]:
    if not SCOPES.is_dir():
        return []
    return sorted(SCOPES.glob("*.scope.json"))


def _load_scope(path: Path) -> dict | None:
    """Read one declaration. None means unreadable, which the caller must treat as refusing."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _staged_paths() -> list[str] | None:
    """Repo-relative paths in the index, including deletions.

    `--diff-filter` is deliberately NOT passed. pre-commit's own staged-file list excludes
    deletions, which is how a delete-only commit slipped past a sibling guard entirely; a
    deletion is a modification of the record, so it belongs in scope like any other.
    """
    out = _git("diff", "--cached", "--name-only")
    if not out:
        # Distinguish "git failed" from "nothing staged": ask git a second, cheap question.
        if _git("rev-parse", "--git-dir") == "":
            return None
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _matches(path: str, patterns: list[str]) -> bool:
    """A path is in scope if it matches a glob, or sits under a declared directory."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        # `--scope docs/notes` is read as "that directory and everything in it", which is what
        # people mean when they type a directory and is otherwise a common false refusal.
        if pattern and not any(ch in pattern for ch in "*?[") and path.startswith(f"{pattern.rstrip('/')}/"):
            return True
    return False


def _scope_self_staged(scope_path: Path, staged: list[str]) -> bool:
    """Is the governing declaration itself being committed?

    If it is, the declaration proves nothing: an agent that edits its own scope in the same
    commit could have written the scope to match whatever it ended up doing.
    """
    try:
        rel = scope_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return rel in staged


def declare(globs: list[str], unseal: list[str], run_id: str) -> int:
    """Record the intended blast radius and the seal, before the work starts."""
    if not globs:
        print("harness-integrity: --declare needs at least one --scope pattern.")
        return 1
    unknown = [p for p in unseal if p not in SEALED_PATHS]
    if unknown:
        # Refuse rather than silently accept: an --unseal for a path that is not sealed is a
        # typo or a stale assumption, and either way the agent believes it has permission it
        # does not have.
        print("harness-integrity: --unseal names paths that are not sealed:")
        for path in unknown:
            print(f"  {path}")
        print("\nSealed paths are:")
        for path in SEALED_PATHS:
            print(f"  {path}")
        return 1

    seals = {}
    for rel in SEALED_PATHS:
        if rel in unseal:
            continue
        digest = _sha256(REPO / rel)
        # A sealed path that does not exist is recorded as absent, so its later APPEARANCE is
        # itself drift -- adding a conftest.py is as much a harness change as editing one.
        seals[rel] = digest

    SCOPES.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "declared_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "scope": list(globs),
        "unsealed": sorted(unseal),
        "seals": seals,
        "why_this_blocks_commits": (
            "This session declared the paths it intended to touch. A staged path outside that "
            "declaration, or a change to a sealed harness file that was not named in it, "
            "refuses the commit."
        ),
    }
    out = SCOPES / f"{run_id}.scope.json"
    out.write_text(json.dumps(record, indent=1) + "\n")

    print(f"harness-integrity: scope declared, run id {run_id}")
    for pattern in globs:
        print(f"  scope   {pattern}")
    for path in sorted(unseal):
        print(f"  unseal  {path}")
    print(f"  sealed  {len(seals)} harness file(s)")
    print(f"\nRelease it when the work is done:\n  python3 {Path(__file__).name} --release --run-id {run_id}")
    return 0


def check() -> int:
    """The pre-commit gate. Inert with no declaration; fail-closed inside one."""
    scopes = _scope_files()
    if not scopes:
        return 0

    staged = _staged_paths()
    if staged is None:
        print("harness-integrity: REFUSING -- could not read the index (git failed).")
        print("  A guard that reports clean when git is broken is worse than no guard.")
        return 1

    failures: list[str] = []
    for scope_path in scopes:
        record = _load_scope(scope_path)
        if record is None:
            failures.append(
                f"{scope_path.name}: unreadable declaration. Unreadable is not the same as "
                f"absent, so this refuses. Delete it deliberately if it is stale."
            )
            continue

        run_id = record.get("run_id", scope_path.stem)
        patterns = [p for p in record.get("scope") or [] if isinstance(p, str)]
        if not patterns:
            failures.append(f"{run_id}: declaration has an empty scope; it permits nothing.")
            continue

        if _scope_self_staged(scope_path, staged):
            failures.append(
                f"{run_id}: the declaration itself is staged in this commit. A scope edited "
                f"in the commit it governs proves nothing -- commit it separately."
            )

        out_of_scope = [p for p in staged if not _matches(p, patterns)]
        if out_of_scope:
            failures.append(
                f"{run_id}: {len(out_of_scope)} staged path(s) outside the declared scope:\n"
                + "\n".join(f"      {p}" for p in out_of_scope)
                + "\n    declared scope:\n"
                + "\n".join(f"      {p}" for p in patterns)
            )

        seals = record.get("seals") or {}
        drifted = []
        for rel, expected in seals.items():
            actual = _sha256(REPO / rel)
            if actual != expected:
                if expected is None:
                    drifted.append(f"{rel} (was absent at declaration, now present)")
                elif actual is None:
                    drifted.append(f"{rel} (was present at declaration, now deleted)")
                else:
                    drifted.append(rel)
        if drifted:
            failures.append(
                f"{run_id}: {len(drifted)} sealed harness file(s) changed in the WORKING TREE "
                f"since this scope was declared:\n"
                + "\n".join(f"      {p}" for p in drifted)
                + "\n    These decide the verdict on other files, so editing one while working "
                "on something else\n    is how a change makes itself pass. To edit one "
                "deliberately, re-declare naming it:\n"
                + "".join(f"      --unseal {p.split(' ')[0]}\n" for p in drifted).rstrip()
            )

    if not failures:
        return 0

    print("harness-integrity: REFUSING THE COMMIT.")
    # Count VIOLATIONS, not scopes: one declaration can break several rules at once, and
    # "2 scopes were violated" when only one exists sends the reader looking for a second
    # session that is not there.
    print(f"  {len(failures)} violation(s) across {len(scopes)} declared scope(s).\n")
    for item in failures:
        print(f"  - {item}\n")
    print("  Fix: stage only what you declared, or re-declare with the wider scope BEFORE")
    print("  doing the work. Releasing a scope to get past this is a decision, not a repair:")
    print(f"    python3 {Path(__file__).name} --release --run-id <id>")
    return 1


def release(run_id: str | None) -> int:
    """Retire one declaration, or report what is still active."""
    scopes = _scope_files()
    if not scopes:
        print("harness-integrity: no active scope.")
        return 0
    if run_id:
        target = SCOPES / f"{run_id}.scope.json"
        if not target.exists():
            print(f"harness-integrity: no scope with run id {run_id}.")
            return 1
        target.unlink()
        print(f"harness-integrity: released {run_id}.")
        return 0
    if len(scopes) > 1:
        # Never guess which one: releasing another session's scope silently removes its guard.
        print("harness-integrity: several scopes are active; name one with --run-id.")
        for path in scopes:
            print(f"  {path.stem.replace('.scope', '')}")
        return 1
    scopes[0].unlink()
    print(f"harness-integrity: released {scopes[0].stem.replace('.scope', '')}.")
    return 0


def show() -> int:
    scopes = _scope_files()
    if not scopes:
        print("harness-integrity: no active scope (the check is inert).")
        return 0
    for path in scopes:
        record = _load_scope(path)
        if record is None:
            print(f"{path.name}: UNREADABLE (this refuses every commit until resolved)")
            continue
        print(f"{record.get('run_id')}  declared {record.get('declared_at')}")
        for pattern in record.get("scope") or []:
            print(f"  scope   {pattern}")
        for rel in record.get("unsealed") or []:
            print(f"  unseal  {rel}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--declare", action="store_true", help="record the intended scope")
    mode.add_argument("--check", action="store_true", help="the pre-commit gate")
    mode.add_argument("--release", action="store_true", help="retire a declaration")
    mode.add_argument("--list", action="store_true", help="show active declarations")
    parser.add_argument("--scope", action="append", default=[], help="a glob the commit may touch")
    parser.add_argument(
        "--unseal", action="append", default=[],
        help="a sealed harness path this scope is permitted to change (exact path, no globs)",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if args.declare:
        return declare(args.scope, args.unseal, _resolve_run_id(args.run_id))
    if args.check:
        return check()
    if args.release:
        return release(args.run_id or os.environ.get(RUN_ID_ENV))
    return show()


if __name__ == "__main__":
    raise SystemExit(main())
