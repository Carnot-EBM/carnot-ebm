#!/usr/bin/env python3
"""Pre-push guard: refuse to publish an un-approved blog post.

WHY THIS EXISTS (2026-06-05 incident)
-------------------------------------
The Carnot blog is operator-curated: per CLAUDE.md "Public Documentation
Discipline" and "Operator-Only External Publication", a blog POST goes live
only when the operator approves it. carnot-ebm.org is served by GitHub Pages
off `main`, so *pushing main is publishing*.

On 2026-06-05 an outer-loop session committed a finished blog draft
(`energy-scorer-not-generator.html`) to `main` "for review", then resumed the
research conductor. The conductor's normal milestone-close auto-push
(`git push origin main`, no operator in the loop) shipped the draft to Pages a
few minutes BEFORE the operator said "approved". The end state matched the
approval, but the operator-only gate was bypassed by an automated push. Root
cause: "commit the draft to main for review" + "main auto-deploys" cannot both
hold. See `ops/known-issues.md` (Blog publish gate) and the memory
`feedback_blog_draft_branch_not_main`.

WHAT THIS GUARD DOES
--------------------
Runs as a git `pre-push` hook (wired via .pre-commit-config.yaml, stage
`pre-push`; also works as a raw .git/hooks/pre-push). For any push whose target
is `refs/heads/main` (or any `main`), it computes the set of blog POST files
(`docs/blog/*.html`, excluding the `index.html` listing page) that are ADDED or
MODIFIED in the pushed range, and blocks the push if any of them is NOT on the
operator-approved allowlist `docs/blog/published-allowlist.txt`.

So: a new post file that the operator has not added to the allowlist cannot be
pushed to main by anyone (conductor, outer-loop, or operator) — the push is
refused with an explicit message. Approving a post is a one-line, deliberate,
operator action: add its filename to the allowlist and commit that.

DESIGN CHOICES
--------------
- `index.html` (the listing/index of posts) is EXEMPT. It is edited routinely
  when posts are added/reordered; the load-bearing artifact is the post file
  itself, which is what the gate protects. A post whose file is not pushed is
  not readable regardless of an index link.
- FAIL-OPEN on *infrastructure* errors (cannot compute the diff — shallow
  clone, missing object, git hiccup): we log loudly to stderr and allow the
  push, so a transient git error never wedges the conductor's push loop.
  FAIL-CLOSED only on a *confirmed* unapproved-post detection. The real-world
  protection comes from the companion convention (drafts live on a branch,
  never on main), so in normal operation main never carries an unapproved post
  and this guard never trips.
- Approve = add the basename (e.g. `my-post.html`) to
  `docs/blog/published-allowlist.txt`. Comments (`#`) and blank lines ignored.

USAGE
-----
Invoked automatically by git/pre-commit on push. Standalone test:
    PRE_COMMIT_FROM_REF=<old> PRE_COMMIT_TO_REF=<new> \
        python3 scripts/blog_publish_guard.py
Exit 0 = allow push; exit 1 = block push (unapproved blog post detected).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOWLIST = REPO / "docs" / "blog" / "published-allowlist.txt"
BLOG_DIR = "docs/blog/"
EXEMPT = {"docs/blog/index.html"}  # the listing page, not a post
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree object


def _run(args: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _load_allowlist() -> set[str]:
    """Return the set of approved post basenames (e.g. {'foo.html'})."""
    if not ALLOWLIST.exists():
        return set()
    out = set()
    for line in ALLOWLIST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(os.path.basename(line))
    return out


def _iter_push_ranges() -> list[tuple[str, str, str]]:
    """Yield (remote_ref, old_sha, new_sha) for each ref being pushed.

    Prefer the pre-commit env protocol (PRE_COMMIT_FROM_REF / _TO_REF +
    PRE_COMMIT_REMOTE_BRANCH); fall back to the raw git pre-push stdin protocol
    ('<local ref> <local sha> <remote ref> <remote sha>' per line).
    """
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


def _changed_blog_posts(old_sha: str, new_sha: str) -> list[str] | None:
    """Blog POST files added/modified in old..new, or None if undeterminable."""
    if new_sha.strip("0") == "":
        # Deleting a ref — nothing being published.
        return []
    base = EMPTY_TREE if old_sha.strip("0") == "" else old_sha
    rc, out, err = _run([
        "git", "diff", "--name-only", "--diff-filter=AM",
        f"{base}..{new_sha}", "--", f"{BLOG_DIR}*.html",
    ])
    if rc != 0:
        sys.stderr.write(f"[blog-publish-guard] WARN: cannot diff {base}..{new_sha}: {err.strip()}\n")
        return None
    posts = []
    for f in out.splitlines():
        f = f.strip()
        if f and f not in EXEMPT and f.startswith(BLOG_DIR) and f.endswith(".html"):
            posts.append(f)
    return posts


def main() -> int:
    allow = _load_allowlist()
    violations: set[str] = set()
    errored = False

    for remote_ref, old_sha, new_sha in _iter_push_ranges():
        if not _targets_main(remote_ref):
            continue
        changed = _changed_blog_posts(old_sha, new_sha)
        if changed is None:
            errored = True
            continue
        for f in changed:
            if os.path.basename(f) not in allow:
                violations.add(f)

    if violations:
        sys.stderr.write(
            "\n[blog-publish-guard] PUSH REFUSED — unapproved blog post(s) on main.\n"
            "  carnot-ebm.org is served from main via GitHub Pages, so pushing\n"
            "  these would PUBLISH them. Blog posts are operator-approved only\n"
            "  (CLAUDE.md Operator-Only External Publication).\n\n"
            "  Unapproved post(s):\n"
        )
        for f in sorted(violations):
            sys.stderr.write(f"    - {f}\n")
        sys.stderr.write(
            "\n  To approve & publish: add the filename to\n"
            f"    {ALLOWLIST.relative_to(REPO)}\n"
            "  and commit that change, then push again. To keep it unpublished,\n"
            "  move the draft off main onto a branch (see memory\n"
            "  feedback_blog_draft_branch_not_main).\n\n"
        )
        return 1

    if errored:
        sys.stderr.write(
            "[blog-publish-guard] note: could not verify some ranges; allowing push "
            "(fail-open on infra error). Blog posts on main are otherwise gated.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
