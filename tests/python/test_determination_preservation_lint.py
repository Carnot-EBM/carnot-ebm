"""A commit may not silently DROP a fabrication-gate determination from a results artifact.

REQ-ARC-WMTE-5995 / SCENARIOs: determination-strip-is-refused, fail-forward-numbers-are-allowed,
deliberate-clearing-requires-a-note, guard-fires-on-unstaged-changes

WHY THIS EXISTS (2026-07-27). A conductor re-run overwrote seven artifacts in place and dropped
``flagged_adversarial: True`` from all seven (six also lost their corrigendum records). That is
not an ordinary never-prune violation: every consumer of CLAUDE.md's fabrication gate keys off
the field being PRESENT, so losing it RE-ADMITS a quarantined artifact to capstone /
evidence-table / paper-v6 aggregation -- silently, with no human-read diff. All seven still
reported ``1 flagged`` after the overwrite, so the determinations were live, not stale.

THE TEST THAT MATTERS MOST IS `test_guard_fires_on_an_unstaged_strip`. The lint's first draft
listed filenames from ``git diff --cached`` while reading the new side from the working tree, so
an unstaged strip produced an EMPTY file list and the lint printed OK on a tree that had just
lost a determination -- it failed to fire on a faithful replay of its own origin incident. A
guard that cannot detect the thing it was written for is worse than no guard, because it
converts an open problem into a false sense of coverage. That regression is pinned here.

The must-NOT-fire controls are equally load-bearing. The operator's standing directive is
fail-forward ("always committing and never reverting so that we fail forward"), so a re-run
that legitimately changes MEASUREMENTS must pass untouched. A lint that refuses normal work
gets disabled, and then protects nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import determination_preservation_lint as dpl  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo with one committed, flagged artifact.

    Driving the REAL git plumbing rather than mocking it: the origin bug was in which git
    command the lint chose (`--cached` vs `HEAD`), so a mocked git would have reproduced the
    bug rather than caught it.
    """
    r = tmp_path / "repo"
    (r / "results").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    art = r / "results" / "experiment_1_thing.json"
    art.write_text(
        json.dumps(
            {
                "experiment": 1,
                "duration_s": 12.5,
                "auroc": 0.91,
                "flagged_adversarial": True,
                "corrigendum_pending": "DURATION_TOO_SHORT",
                "corrigendum_note": "flagged 2026-05-30",
            },
            indent=2,
        )
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")
    monkeypatch.setattr(dpl, "REPO", r)
    return r, art


def _write(art: Path, **changes) -> None:
    d = json.loads(art.read_text())
    for k, v in changes.items():
        if v is dpl:  # sentinel: delete the key
            d.pop(k, None)
        else:
            d[k] = v
    art.write_text(json.dumps(d, indent=2) + "\n")


def test_stripping_the_stamp_is_refused(repo):
    """THE ORIGIN INCIDENT: the stamp vanishes on an in-place overwrite."""
    _, art = repo
    _write(art, flagged_adversarial=dpl)
    v = dpl.check()
    assert v, "a dropped determination must be refused"
    assert any("flagged_adversarial True ->" in x for x in v)


def test_guard_fires_on_an_unstaged_strip(repo):
    """THE REGRESSION THAT MADE THE FIRST DRAFT USELESS -- pinned.

    The strip is written to the working tree and NOT staged. The first draft listed files from
    `git diff --cached`, found none, and returned clean.
    """
    _, art = repo
    _write(art, flagged_adversarial=dpl)
    assert _git(repo[0], "diff", "--cached", "--name-only") == "", "precondition: nothing staged"
    assert dpl.check(), "the guard must not depend on the change being staged"


def test_losing_a_corrigendum_record_is_refused(repo):
    """The corrigendum trail is the evidence behind the stamp; a re-run does not supersede it."""
    _, art = repo
    _write(art, corrigendum_pending=dpl, corrigendum_note=dpl)
    v = dpl.check()
    assert any("lost corrigendum record" in x for x in v)


def test_changing_measurements_while_keeping_the_stamp_passes(repo):
    """MUST-NOT-FIRE 1: fail-forward. A re-run's new numbers are normal, healthy work."""
    _, art = repo
    _write(art, duration_s=999.9, auroc=0.42, a_new_metric=0.1)
    assert dpl.check() == [], (
        "a lint that refuses normal re-runs gets disabled and protects nothing"
    )


def test_deliberate_clearing_with_a_note_passes(repo):
    """MUST-NOT-FIRE 2: a determination CAN be retracted -- auditably."""
    _, art = repo
    _write(
        art,
        flagged_adversarial=False,
        flagged_adversarial_cleared_note="Cleared: substrate taxonomy fixed; 0 flagged now.",
    )
    assert not [x for x in dpl.check() if "flagged_adversarial True ->" in x]


def test_clearing_without_a_note_is_refused(repo):
    """False-with-no-note is indistinguishable from the accident, so it is refused."""
    _, art = repo
    _write(art, flagged_adversarial=False)
    assert any("no *_cleared_note" in x for x in dpl.check())


def test_an_unflagged_artifact_is_never_implicated(repo):
    """Only artifacts that HELD a determination can lose one."""
    r, _ = repo
    other = r / "results" / "experiment_2_clean.json"
    other.write_text(json.dumps({"experiment": 2, "duration_s": 5.0}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "add clean")
    other.write_text(json.dumps({"experiment": 2, "duration_s": 6.0}, indent=2) + "\n")
    assert dpl.check() == []


def test_unparseable_or_absent_sides_do_not_crash_the_commit(repo):
    """A truncated artifact mid-write must not wedge every commit in the repo."""
    _, art = repo
    art.write_text('{"experiment": 1, "flagged')
    assert dpl.check() == [], "unparseable NEW side is not evidence of a dropped determination"


def test_the_real_repo_is_currently_clean():
    """The live tree must hold every determination it holds at HEAD.

    Guards the 7 restorations from 2026-07-27: if a later re-run strips one again, this fails
    in CI even if the commit-time hook was bypassed.
    """
    assert dpl.check() == [], "a determination has been dropped in the working tree"
