#!/usr/bin/env python3
"""Pre-push guard: refuse to push a stray submodule gitlink (mode 160000) that
has no matching .gitmodules entry.

WHY THIS EXISTS (2026-06-06 incident)
-------------------------------------
A conductor experiment (commit c4c612662, "Latent-Symbol Bridge Task 0") cloned
`nano-trm` into the working tree; the conductor's `git add -A` then committed the
embedded repo as a submodule GITLINK (tree mode 160000) with NO `.gitmodules`
entry. Every GitHub Actions checkout that initializes submodules — the
"pages build and deployment" workflow, CI, the phase1-reproducer — then aborted
with:

    fatal: No url found for submodule path 'nano-trm' in .gitmodules
    The process '/usr/bin/git' failed with exit code 128

So carnot-ebm.org froze on a stale Pages build (a freshly-approved blog post
404'd) and CI went red, for ~37 minutes before it was noticed and fixed. The
class of bug: ANY experiment that clones a repo into the working dir can be
swallowed by `git add -A` as a gitlink and break every checkout the same way,
under a different name. `.gitignore` covers the *known* offenders; this guard
covers the *class*.

WHAT THIS GUARD DOES
--------------------
Runs as part of the standalone git `pre-push` hook (scripts/git-hooks/pre-push).
For any push whose target is `main`, it finds files ADDED or MODIFIED to tree
mode 160000 (a gitlink) in the pushed range and blocks the push if any such path
is NOT declared as a submodule in `.gitmodules` (at the pushed tip). A real,
intentional submodule (with a `.gitmodules` `path =` entry) passes; an accidental
embedded-repo gitlink does not.

It runs at PUSH time on purpose: the conductor commits with `--no-verify`
(skipping commit-stage hooks), but its `git push origin main` does NOT pass
`--no-verify`, so a standalone pre-push hook is the reliable chokepoint. Blocking
the push is the right failure mode: a wedged push that alerts loudly is far
cheaper than propagating a checkout-breaking gitlink to the remote where it takes
down Pages + CI for everyone.

FAIL MODE
---------
FAIL-OPEN on infrastructure errors (cannot compute the diff — shallow clone,
missing object): log to stderr and allow, so a transient git hiccup never wedges
the push loop. FAIL-CLOSED only on a confirmed stray gitlink.

REMEDIATION (printed on block)
------------------------------
    git rm --cached <path>          # untrack the gitlink (keeps it on disk)
    echo '/<path>/' >> .gitignore   # stop git add -A from re-committing it
    git commit -m 'untrack stray gitlink <path>'
…or, if it is meant to be a real submodule, add a `.gitmodules` entry for it.

USAGE
-----
Invoked by scripts/git-hooks/pre-push (reads the git pre-push stdin protocol:
'<local ref> <local sha> <remote ref> <remote sha>' per line). Also honors the
pre-commit env protocol (PRE_COMMIT_FROM_REF/_TO_REF) if present. Standalone test:
    echo 'refs/heads/main <new> refs/heads/main <old>' | python3 scripts/gitlink_guard.py
Exit 0 = allow; exit 1 = block.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
GITLINK_MODE = "160000"


def _run(args: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _iter_push_ranges() -> list[tuple[str, str, str]]:
    frm = os.environ.get("PRE_COMMIT_FROM_REF")
    to = os.environ.get("PRE_COMMIT_TO_REF")
    if frm is not None and to is not None:
        remote_ref = os.environ.get("PRE_COMMIT_REMOTE_BRANCH", "refs/heads/main")
        return [(remote_ref, frm, to)]
    ranges: list[tuple[str, str, str]] = []
    for line in sys.stdin:
        parts = line.split()
        if len(parts) == 4:
            _local_ref, local_sha, remote_ref, remote_sha = parts
            ranges.append((remote_ref, remote_sha, local_sha))
    return ranges


def _targets_main(remote_ref: str) -> bool:
    return remote_ref.split("/")[-1] == "main" or remote_ref == "main"


def _gitmodules_paths(tree_sha: str) -> set[str]:
    """Submodule paths declared in .gitmodules at tree_sha (empty set if none)."""
    rc, out, _ = _run(["git", "show", f"{tree_sha}:.gitmodules"])
    if rc != 0:
        return set()
    paths = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("path") and "=" in line:
            paths.add(line.split("=", 1)[1].strip())
    return paths


def _added_gitlinks(old_sha: str, new_sha: str) -> list[str] | None:
    """Paths added/modified to mode 160000 in old..new, or None if undeterminable."""
    if new_sha.strip("0") == "":
        return []
    base = EMPTY_TREE if old_sha.strip("0") == "" else old_sha
    rc, out, err = _run(["git", "diff", "--raw", "--diff-filter=AM", f"{base}..{new_sha}"])
    if rc != 0:
        sys.stderr.write(f"[gitlink-guard] WARN: cannot diff {base}..{new_sha}: {err.strip()}\n")
        return None
    links = []
    for line in out.splitlines():
        # ':<oldmode> <newmode> <oldsha> <newsha> <status>\t<path>'
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        fields = meta.lstrip(":").split()
        if len(fields) >= 2 and fields[1] == GITLINK_MODE:
            links.append(path.strip())
    return links


def main() -> int:
    violations: set[str] = set()
    errored = False
    for remote_ref, old_sha, new_sha in _iter_push_ranges():
        if not _targets_main(remote_ref):
            continue
        links = _added_gitlinks(old_sha, new_sha)
        if links is None:
            errored = True
            continue
        if not links:
            continue
        declared = _gitmodules_paths(new_sha)
        for path in links:
            if path not in declared:
                violations.add(path)

    if violations:
        sys.stderr.write(
            "\n[gitlink-guard] PUSH REFUSED — stray submodule gitlink(s) with no .gitmodules entry.\n"
            "  A gitlink (tree mode 160000) with no matching .gitmodules entry breaks\n"
            "  EVERY GitHub Actions checkout that inits submodules (Pages, CI, ...):\n"
            "    fatal: No url found for submodule path '<path>' in .gitmodules\n"
            "  This is almost always an experiment repo accidentally swallowed by\n"
            "  `git add -A`. Offending path(s):\n"
        )
        for p in sorted(violations):
            sys.stderr.write(f"    - {p}\n")
        sys.stderr.write(
            "\n  Fix (untrack it, keep it on disk, stop it recurring):\n"
            "    git rm --cached <path>\n"
            "    echo '/<path>/' >> .gitignore\n"
            "    git commit -m 'untrack stray gitlink <path>'\n"
            "  …or, if it is a real submodule, add a .gitmodules entry for it.\n\n"
        )
        return 1

    if errored:
        sys.stderr.write(
            "[gitlink-guard] note: could not verify some ranges; allowing push "
            "(fail-open on infra error).\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
