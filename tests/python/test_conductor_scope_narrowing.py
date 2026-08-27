"""REQ-INFRA-6800: the conductor stops sweeping work another session has claimed.

INCIDENT 2026-08-27. The conductor commits with `git add -A`, deliberately -- it preserves
in-flight work rather than losing it, which is the standing commit-first discipline. But it
also swept OTHER sessions' uncommitted work into conductor-attributed commits four times in
one day, once taking a guard while that guard was being written. One agent then searched the
object store, concluded a colleague's work had been destroyed, and offered to rewrite work
that had in fact survived.

The narrowing is BY CLAIM, not by pathspec, and the distinction is the whole design. A fixed
pathspec is the dangerous version: agent-authored files outside it -- new tests, spec edits,
experiment modules -- would be committed by nobody and left exposed to the next checkout. So
everything is still staged, and only paths another session has explicitly declared are
unstaged, which by definition means someone else is going to commit them.

The decision rule is the pure function `claimed_by_other_sessions`, tested here without a git
fixture, following the precedent of `determination_damage`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from research_conductor import claimed_by_other_sessions  # noqa: E402


def test_an_exact_path_claim_is_honoured() -> None:
    claimed = claimed_by_other_sessions(
        ["scripts/foo.py", "results/bar.json"], {"peer": ["scripts/foo.py"]}
    )
    assert claimed == {"scripts/foo.py": "peer"}


def test_unclaimed_work_is_never_excluded() -> None:
    """The property that makes this safe: nobody's work is stranded.

    A path no session has declared is still swept up exactly as before, so narrowing cannot
    leave a file that will be committed by nobody.
    """
    staged = ["results/a.json", "ops/status.md", "python/carnot/new_module.py"]
    assert claimed_by_other_sessions(staged, {"peer": ["scripts/foo.py"]}) == {}


def test_a_directory_claim_covers_its_contents() -> None:
    """`--scope docs/notes` means the directory, matching harness_integrity_lint._matches.

    The two must agree about what a declaration covers, or a session would be refused a
    commit for a path the conductor had already taken.
    """
    claimed = claimed_by_other_sessions(
        ["docs/notes/a.md", "docs/notes/deep/b.md", "docs/other.md"],
        {"peer": ["docs/notes"]},
    )
    assert claimed == {"docs/notes/a.md": "peer", "docs/notes/deep/b.md": "peer"}


def test_a_glob_claim_is_honoured() -> None:
    claimed = claimed_by_other_sessions(
        ["scripts/a.py", "scripts/b.py", "ops/c.md"], {"peer": ["scripts/*.py"]}
    )
    assert set(claimed) == {"scripts/a.py", "scripts/b.py"}


def test_the_first_claiming_session_wins_and_each_path_has_one_owner() -> None:
    """Two sessions claiming the same path must not produce an ambiguous owner."""
    claimed = claimed_by_other_sessions(
        ["scripts/foo.py"], {"first": ["scripts/foo.py"], "second": ["scripts/*"]}
    )
    assert list(claimed) == ["scripts/foo.py"]
    assert claimed["scripts/foo.py"] in {"first", "second"}


def test_no_claims_means_nothing_is_excluded() -> None:
    """With no declarations the behaviour is byte-for-byte the old `git add -A`."""
    assert claimed_by_other_sessions(["a", "b"], {}) == {}


def test_a_greedy_claim_is_detected_so_the_caller_can_refuse_it() -> None:
    """A session declaring `*` claims everything.

    The pure rule reports that honestly; the caller's safety valve is what refuses to act on
    it, because turning the conductor's checkpoint into an empty commit would lose the very
    work the sweep exists to preserve.
    """
    staged = ["a.py", "b.json", "c.md"]
    claimed = claimed_by_other_sessions(staged, {"greedy": ["*"]})
    assert len(claimed) == len(staged)


def test_a_claim_never_matches_a_partial_directory_name() -> None:
    """`docs/note` must not claim `docs/notes/a.md`."""
    claimed = claimed_by_other_sessions(["docs/notes/a.md"], {"peer": ["docs/note"]})
    assert claimed == {}
