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
from datetime import UTC, datetime, timedelta
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
# WHAT IS ACTUALLY SEALED IN PRACTICE, and it is not this list (2026-08-29).
# An explicit `--unseal` in ANY active declaration lifts that path for every session, and the
# conductor's standing declaration unseals four of the nine below -- .pre-commit-config.yaml,
# adversarial_verify.py, operator_curated_docs_lint.py and conftest.py -- because it AUTHORS
# them as normal task work (measured: 90 authored commits to adversarial_verify.py alone).
# So while the conductor's scope exists, the effective sealed set is FIVE files and the
# fabrication gate is NOT among them. That is deliberate -- sealing a file the loop edits
# hourly would wedge it -- but a reader of this tuple would otherwise believe nine are sealed.
# Run `--list` to see what is actually lifted right now.
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


#: Stored in place of a hash when a declaration is anchored to HEAD instead of to the moment
#: it was made. See `--seal-anchor head`.
HEAD_ANCHOR = "@HEAD"


def _head_blob(rel: str) -> tuple[str | None, bool]:
    """(blob id at HEAD, ok). `ok` is False when git could not answer at all.

    `git ls-tree` is used rather than `rev-parse HEAD:<path>` because rev-parse exits
    non-zero both when git is broken and when the path simply is not in HEAD -- and those
    must not be confused. ls-tree exits 0 either way and prints nothing for an absent path.
    """
    out = _git("ls-tree", "HEAD", "--", rel)
    if out == "":
        # Empty means "absent at HEAD" only if git is answering at all; ask once more.
        if _git("rev-parse", "--git-dir") == "":
            return None, False
        return None, True
    fields = out.split()
    return (fields[2] if len(fields) >= 3 else None), True


def _worktree_blob(rel: str) -> tuple[str | None, bool]:
    """(blob id the working-tree file would hash to, ok). None with ok=True means absent."""
    path = REPO / rel
    if not path.exists():
        return None, True
    out = _git("hash-object", "--", rel)
    if out == "":
        return None, False
    return out.strip(), True


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
        if (
            pattern
            and not any(ch in pattern for ch in "*?[")
            and path.startswith(f"{pattern.rstrip('/')}/")
        ):
            return True
    return False


#: How long a declaration may claim a path before it is treated as abandoned.
#:
#: FAIL TOWARD NOT BLOCKING, deliberately. A wrongly-held claim stops real work and there is
#: nobody to appeal to; a wrongly-released one costs attribution on one commit. Agents die
#: without releasing their scope routinely -- a killed run leaves its declaration on disk
#: forever -- so without expiry the first crashed session would permanently lock its paths.
CLAIM_STALE_AFTER = timedelta(hours=4)


def is_universal_pattern(pattern: str) -> bool:
    """Does this pattern claim the whole repository rather than a part of it?

    THIS IS WHAT KEEPS ENFORCEMENT FROM DEADLOCKING EVERYTHING. The conductor holds a STANDING
    declaration of `*` because it sweeps all uncommitted work -- that is a statement about what
    it will COMMIT, not a claim that no other session may touch anything. Enforce `*` literally
    and no agent could ever stage a file again.

    So a pattern with no literal path content is advisory: it still governs what its own owner
    may stage, and it never blocks a stranger. A pattern naming any real path segment is
    enforceable.
    """

    stripped = pattern.strip()
    if not stripped:
        return True
    if all(ch in "*?/[]!-" for ch in stripped):
        return True
    # A BARE EXTENSION CLAIMS THE WHOLE REPOSITORY (2026-08-29). `fnmatch`'s `*` crosses `/`,
    # so `--scope '*.py'` is a one-line declaration that owns every Python file in the tree --
    # `python/carnot/paths.py`, `tests/python/conftest.py`, the conductor itself -- against
    # every other session for the whole staleness window. It "names a real path segment" only
    # in the sense of an extension, so the first version classified it enforceable.
    #
    # The concept is "claims the whole repository in practice", and a pattern with no directory
    # separator that begins with a wildcard satisfies it. `scripts/*.py` is unaffected: it names
    # a directory and stays enforceable.
    if "/" not in stripped and stripped.startswith(("*", "?")):
        return True
    return False


def claim_instant(record: dict) -> datetime | None:
    """When this declaration was made, as an aware UTC datetime. None means UNUSABLE.

    ONE PARSE, ONE CLOCK. The first version keyed ordering on the RAW string and expiry on a
    parsed value, so the two disagreed in three ways an adversarial review demonstrated:

      * `"!bad-timestamp"` is unparseable, so expiry treated it as fresh forever, while `"!"`
        (0x21) sorts below every real ISO stamp -- one corrupt file owned its patterns
        permanently. The comment claimed an unusable timestamp "loses every comparison"; it
        won them all. A pattern list narrower than the concept it named.
      * `"...10:00:00Z"` versus `"...10:00:00.500000+00:00"`: `.` (0x2E) < `Z` (0x5A), so the
        stamp half a second LATER sorted older.
      * `"08:00:00-05:00"` is 13:00Z and sorted before `"09:00:00Z"`.

    Only the first is reachable today, because `declare()` is the sole writer and always emits
    `+00:00` with microseconds. The other two are one hand-edit or one second writer away, and
    a comparison that is right only while one function has a monopoly on the format is not
    right.
    """

    raw = record.get("declared_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _claim_age_key(record: dict) -> tuple[int, datetime]:
    """Sortable age. An unusable timestamp sorts after every real one, so it never wins."""
    instant = claim_instant(record)
    return (1, datetime.max.replace(tzinfo=UTC)) if instant is None else (0, instant)


def _claim_is_stale(record: dict, now: datetime) -> bool:
    """Is this claim abandoned? An UNUSABLE timestamp counts as stale.

    Treating an undated claim as fresh-forever was the other half of the corrupt-record bug:
    it could neither be aged out nor out-ranked. A claim that cannot say when it was made
    cannot be trusted to hold a path against a claim that can.
    """
    instant = claim_instant(record)
    if instant is None:
        return True
    return (now - instant) > CLAIM_STALE_AFTER


def claim_owner(path: str, records: list[dict], now: datetime) -> str | None:
    """Which session owns this path? None means nobody, so anyone may stage it.

    WOUND-WAIT ORDERING. The OLDEST declaration wins; ties break on run id. Both properties
    matter and for different reasons.

    Oldest-wins makes the resolution deadlock-free by construction: the order is total and
    time-based, so a cycle cannot form. We reached a real deadlock on 2026-08-29 with the
    previous first-come rule -- two narrow scopes each refusing the other's commit -- and worked
    around it by removing cross-session judging entirely. This restores the judging with an
    ordering that cannot deadlock, rather than leaving it off.

    Tie-breaking on run id makes it DETERMINISTIC: every session computes the same owner from
    the same files, so the wounded side is told the same thing the winner believes. A rule where
    each side thinks it won is worse than no rule.

    The wounded session keeps its edits and re-declares. We wound the CLAIM, never the work --
    Fission can treat an abort as an early commit because its writes are prefix-safe, and a
    half-applied source edit is not.
    """

    best: tuple[str, str] | None = None
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        if _claim_is_stale(record, now):
            continue
        patterns = [
            p
            for p in record.get("scope") or []
            if isinstance(p, str) and not is_universal_pattern(p)
        ]
        if not patterns or not _matches(path, patterns):
            continue
        key = (_claim_age_key(record), run_id)
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


def _literals(patterns: list[str]) -> list[str]:
    """Concrete paths usable as probes: the patterns that name a path rather than a shape."""
    return [p for p in patterns if not any(ch in p for ch in "*?[")]


def overlapping_older_claims(
    globs: list[str], mine: dict, others: list[dict], now: datetime
) -> dict[str, str]:
    """My patterns that an OLDER live declaration also covers, mapped to that owner.

    Two glob patterns cannot be intersected exactly in the general case, so this probes with
    the CONCRETE paths each side named, in both directions: their literals against my patterns,
    and my literals against theirs. A first version probed by substituting "x" for the
    wildcards, which found nothing at all -- `scripts/*` became `scripts/x`, and the literal
    `scripts/target.py` it genuinely collides with does not match that.

    Being approximate here is acceptable in a way it would not be in `claim_owner`: a missed
    overlap costs a warning the author does not get, and the commit-time refusal still holds.
    """

    found: dict[str, str] = {}
    for pattern in globs:
        if is_universal_pattern(pattern):
            continue
        for other in others:
            if _claim_is_stale(other, now):
                continue
            owner = other.get("run_id")
            if not isinstance(owner, str) or not owner:
                continue
            if _claim_age_key(other) >= _claim_age_key(mine):
                continue  # not older, so it does not win
            theirs = [
                q
                for q in other.get("scope") or []
                if isinstance(q, str) and not is_universal_pattern(q)
            ]
            if not theirs:
                continue
            # Literal probes cannot see glob-vs-glob overlap, and two sessions declaring the
            # SAME glob is the single most likely collision there is -- it produced no warning
            # at all, so the younger session learned at commit time, after the work, which is
            # the exact cost this warning exists to avoid. Exact equality and mutual matching
            # cover the common cases; full glob intersection is undecidable in general and the
            # commit-time refusal remains the backstop.
            overlap = (
                pattern in theirs
                or any(_matches(lit, [pattern]) for lit in _literals(theirs))
                or any(_matches(lit, theirs) for lit in _literals([pattern]))
                or any(fnmatch.fnmatch(q, pattern) or fnmatch.fnmatch(pattern, q) for q in theirs)
            )
            if overlap:
                found[pattern] = owner
                break
    return found


def contested_paths(
    staged: list[str], records: list[dict], committer: str | None, now: datetime
) -> dict[str, str]:
    """Staged paths another session owns, mapped to that owner.

    When the committer cannot be identified, enforcement is SKIPPED rather than applied to all.
    That is the opposite of the seal rule directly above, and the asymmetry is intended: an
    unidentified session held to every claim could not commit at all while any claim stood,
    which turns a missing environment variable into a repository-wide freeze.
    """

    if committer is None:
        return {}
    owned: dict[str, str] = {}
    for path in staged:
        owner = claim_owner(path, records, now)
        if owner is not None and owner != committer:
            owned[path] = owner
    return owned


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


def _prior_declared_at(run_id: str) -> str | None:
    """The timestamp this run id first declared with, if it is still live and usable.

    A STALE prior returns None, so an abandoned claim that is picked up again starts a fresh
    clock rather than resurrecting expired seniority.
    """
    path = SCOPES / f"{run_id}.scope.json"
    record = _load_scope(path)
    if record is None or _claim_is_stale(record, datetime.now(UTC)):
        return None
    value = record.get("declared_at")
    return value if isinstance(value, str) and value else None


def declare(globs: list[str], unseal: list[str], run_id: str, anchor: str = "declaration") -> int:
    """Record the intended blast radius and the seal, before the work starts.

    `anchor` chooses what the seal is measured against. "declaration" hashes each sealed file
    now, which is right for a short scoped task. "head" records a sentinel and compares the
    working tree to HEAD at check time, which is right for a STANDING declaration that will
    outlive many commits -- a declaration-time hash would go stale the first time anyone
    legitimately committed a harness change and then refuse every commit until a human
    noticed.
    """
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
        if anchor == "head":
            seals[rel] = HEAD_ANCHOR
            continue
        digest = _sha256(REPO / rel)
        # A sealed path that does not exist is recorded as absent, so its later APPEARANCE is
        # itself drift -- adding a conftest.py is as much a harness change as editing one.
        seals[rel] = digest

    SCOPES.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        # SENIORITY SURVIVES A RE-DECLARE (2026-08-29). Rewriting this unconditionally meant a
        # session working past the staleness window lost its own declared paths to any younger
        # claimant, and could never regain them: the refusal's own advice -- re-declare -- reset
        # the clock, so it stayed junior forever. An expiry meant to release ABANDONED claims was
        # demoting live ones. Re-declaring the same run id keeps its original timestamp, so the
        # record answers "when did this session first claim this" rather than "when did it last
        # type the command".
        "declared_at": _prior_declared_at(run_id) or datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "scope": list(globs),
        "unsealed": sorted(unseal),
        "seal_anchor": anchor,
        "seals": seals,
        "why_this_blocks_commits": (
            "This session declared the paths it intended to touch. A staged path outside that "
            "declaration, or a change to a sealed harness file that was not named in it, "
            "refuses the commit."
        ),
    }
    out = SCOPES / f"{run_id}.scope.json"
    out.write_text(json.dumps(record, indent=1) + "\n")

    # Tell the declaring session NOW if an older claim already overlaps what it just asked for.
    # Discovering it at commit time means the work is already done; discovering it here costs
    # nothing. This warns rather than refuses -- overlapping intent is often legitimate (the
    # older session may be about to release) and a declaration that cannot be made is worse
    # than one made with open eyes.
    now = datetime.now(UTC)
    others = [
        r for r in (_load_scope(s) for s in _scope_files()) if r and r.get("run_id") != run_id
    ]
    wounds = overlapping_older_claims(globs, record, others, now)
    if wounds:
        print("harness-integrity: WOUNDED -- an older declaration already claims:")
        for pattern, owner in sorted(wounds.items()):
            print(f"  {pattern}  overlaps {owner}")
        print(
            "  Commits touching those paths will be refused while that claim stands.\n"
            "  Keep your edits; narrow this scope, or wait for that session to release."
        )

    print(f"harness-integrity: scope declared, run id {run_id}")
    for pattern in globs:
        print(f"  scope   {pattern}")
    for path in sorted(unseal):
        print(f"  unseal  {path}")
    print(f"  sealed  {len(seals)} harness file(s), anchored to {anchor}")
    # EXPORT THIS OR NOTHING WORKS. `check()` identifies the committing session solely by this
    # variable. Until 2026-08-29 nothing in the repository ever set it -- a grep found exactly
    # one reference, its own definition -- so every committer read as unidentified: cross-session
    # enforcement never fired for anyone, while the out-of-scope branch judged the unidentified
    # committer by EVERY declaration and refused it. The feature was dead and the freeze was
    # live, which is the worst pairing of the two policies.
    print("\nExport this so your commits are recognised as yours:")
    print(f"  export {RUN_ID_ENV}={run_id}")
    print(
        f"\nRelease it when the work is done:\n  python3 {Path(__file__).name} --release --run-id {run_id}"
    )
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

    # A path explicitly named by ANY active declaration is unsealed for this commit.
    #
    # Without this, a STANDING declaration (the conductor's) deadlocks harness work: it seals
    # the record-preservation lints, and the person legitimately fixing one cannot unseal it,
    # because their own --unseal does not reach another declaration's seal. They would have to
    # release someone else's scope to commit, which is exactly the "release it to get past
    # this" move the refusal text warns against.
    #
    # This does let an agent bypass a standing seal by naming the path. That is deliberate and
    # consistent with the rest of the design: a glob still never unseals, so the bypass costs
    # a typed path name and leaves the intent in the record, which is the bar the seal sets.
    explicitly_unsealed: set[str] = set()
    for scope_path in scopes:
        record = _load_scope(scope_path)
        if record:
            explicitly_unsealed.update(record.get("unsealed") or [])

    judged_at = datetime.now(UTC)
    live_records = [
        r for r in (_load_scope(s) for s in scopes) if r and not _claim_is_stale(r, judged_at)
    ]

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

        # WHOSE scope judges this commit (2026-08-29). Out-of-scope is judged against the
        # COMMITTING session's declaration only. Before this, every active declaration judged
        # every commit, so two narrow scopes deadlocked each other: agent A declares a.py,
        # agent B declares b.py, B stages its own b.py, and A's record refuses it. Demonstrated
        # in a scratch repo by an independent review, and it bit that reviewer live. It was
        # masked only because the sole standing declaration is '*', which matches everything.
        #
        # SEALS ARE UNAFFECTED and still apply across every declaration -- a harness file is
        # sealed against everyone, which is the property worth having. Only the "did YOU stage
        # something you did not declare" question becomes per-session.
        #
        # When the committing session cannot be identified, EVERY declaration judges, which is
        # the old behaviour and the strict direction: an unidentified committer is held to all
        # of them rather than none.
        # WHO JUDGES AN UNIDENTIFIED COMMITTER (corrected 2026-08-29). This read
        # `committer is None or run_id == committer` -- every declaration judged every
        # unattributable commit. Combined with nothing ever setting the variable, that made the
        # 2026-08-29 deadlock still live in the deployed state: agent A declares a.py, agent B
        # declares b.py, B stages only its own b.py, and A's record refuses it. Reproduced
        # exactly, following the tool's own documented usage.
        #
        # An unattributable commit cannot be checked against "did YOU stage something you did
        # not declare" -- there is no YOU. Guessing it belongs to all of them freezes the repo.
        # The one case where attribution is unambiguous is a single live declaration, so that
        # is the only case where an unidentified committer is judged.
        #
        # STALENESS APPLIES HERE TOO. It did not, and the refusal text promising that a claim
        # older than the window "stops blocking on its own" was false for the branch that
        # actually fired: a crashed agent locked every other session out permanently.
        committer = os.environ.get(RUN_ID_ENV)
        if _claim_is_stale(record, judged_at):
            judges_scope = False
        elif committer is not None:
            judges_scope = run_id == committer
        else:
            judges_scope = len(live_records) == 1
        out_of_scope = [p for p in staged if not _matches(p, patterns)] if judges_scope else []
        if out_of_scope:
            failures.append(
                f"{run_id}: {len(out_of_scope)} staged path(s) outside the declared scope:\n"
                + "\n".join(f"      {p}" for p in out_of_scope)
                + "\n    declared scope:\n"
                + "\n".join(f"      {p}" for p in patterns)
            )

        seals = record.get("seals") or {}
        # A DEAD SESSION'S DECLARATION-ANCHORED SEAL MUST NOT FREEZE THE REPO FOREVER
        # (2026-08-29). Staleness was applied to the out-of-scope branch and not to this one,
        # one branch over. A crashed declaration refuses every commit repo-wide the moment any
        # sealed file legitimately changes, and only deleting the file recovers it.
        #
        # HEAD-ANCHORED SEALS ARE DELIBERATELY EXEMPT. `--seal-anchor head` marks a STANDING
        # declaration -- the conductor's, which is re-asserted at startup and is routinely
        # older than the window; it was 6.5h old when this was written. Its seal is also
        # self-limiting: it compares the working tree to HEAD, so committing the change clears
        # it. Expiring those would silently remove the only harness protection in the repo,
        # which is the failure this rule is supposed to prevent, achieved by the fix for it.
        if seals and record.get("seal_anchor") != "head" and _claim_is_stale(record, judged_at):
            seals = {}
        drifted = []
        for rel, expected in seals.items():
            if rel in explicitly_unsealed:
                continue
            if expected == HEAD_ANCHOR:
                # Anchored to HEAD, not to the declaration. A standing declaration -- one that
                # outlives many commits, like the conductor's -- would otherwise go stale the
                # first time anyone legitimately commits a harness change, and then refuse
                # every commit until a human noticed. Anchored here, a committed change moves
                # the baseline with it and only UNCOMMITTED harness edits refuse.
                head, ok_head = _head_blob(rel)
                work, ok_work = _worktree_blob(rel)
                if not (ok_head and ok_work):
                    drifted.append(f"{rel} (git could not be asked; refusing rather than guessing)")
                elif head != work:
                    if head is None:
                        drifted.append(f"{rel} (not in HEAD, present uncommitted)")
                    elif work is None:
                        drifted.append(f"{rel} (in HEAD, deleted uncommitted)")
                    else:
                        drifted.append(f"{rel} (modified, uncommitted)")
                continue
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

    # ENFORCED CLAIMS (2026-08-29). Until now a declaration bound only its own author: it
    # constrained what THAT session could stage and did nothing about a stranger staging the
    # same path. That is advisory in exactly the direction that hurt -- a shared index took one
    # agent's staged work into another's commit three times in one day, and once took this very
    # guard while it was being written.
    #
    # Now a path owned by another live session refuses the commit, with the owner named so the
    # two can be told apart. Ownership is wound-wait ordered (see `claim_owner`), so conflicting
    # claims resolve the same way from either side instead of deadlocking.
    committer_id = os.environ.get(RUN_ID_ENV)
    records = [r for r in (_load_scope(s) for s in scopes) if r]
    contested = contested_paths(staged, records, committer_id, judged_at)
    if contested:
        by_owner: dict[str, list[str]] = {}
        for path, owner in sorted(contested.items()):
            by_owner.setdefault(owner, []).append(path)
        detail = []
        for owner, paths in sorted(by_owner.items()):
            detail.append(f"claimed by {owner}:\n" + "\n".join(f"      {p}" for p in paths))
        failures.append(
            f"{len(contested)} staged path(s) belong to another session's declaration.\n    "
            + "\n    ".join(detail)
            + "\n    Your edits are safe -- this refuses the COMMIT, not the work. Either wait "
            "for that\n    session to finish and release, or re-declare without these paths and "
            "commit the rest.\n    A claim older than "
            f"{int(CLAIM_STALE_AFTER.total_seconds() // 3600)}h is treated as abandoned and "
            "stops blocking on its own."
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
        "--unseal",
        action="append",
        default=[],
        help="a sealed harness path this scope is permitted to change (exact path, no globs)",
    )
    parser.add_argument(
        "--seal-anchor",
        choices=("declaration", "head"),
        default="declaration",
        help="measure seals against the moment of declaration (default) or against HEAD "
        "(for a standing declaration that will outlive many commits)",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if args.declare:
        return declare(args.scope, args.unseal, _resolve_run_id(args.run_id), args.seal_anchor)
    if args.check:
        return check()
    if args.release:
        return release(args.run_id or os.environ.get(RUN_ID_ENV))
    return show()


if __name__ == "__main__":
    raise SystemExit(main())
