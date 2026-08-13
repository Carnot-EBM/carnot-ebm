"""The conductor's determination restore must catch DOWNGRADE, not just DELETION.

INCIDENT 2026-08-12. Five artifacts reached the git index with `flagged_adversarial`
changed True -> None, lifting their quarantine. `_restore_dropped_determinations` already
existed and ran before every conductor `git add -A`, and it restored NOTHING, because its
test was `k not in cur` -- which is False when the key is present and merely nulled. The
damage was caught only because `determination-preservation-lint` refused a HUMAN commit, and
that lint never runs on conductor commits (`--no-verify`, deliberately, for anti-stash-loss
reasons documented in `git_commit_and_push`).

This is the SILENT_NON_FIRING class named in CLAUDE.md's QA-Layer Authenticity Discipline:
a guard whose pattern is narrower than the concept it claims to protect. These tests pin
both damage shapes and the deliberate-clear carve-out.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _restore_in(tmp_path: Path, head_obj: dict, cur_obj: dict) -> dict:
    """Run the real helper against a throwaway git repo and return the resulting artifact."""
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, check=False)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    results = tmp_path / "results"
    results.mkdir()
    art = results / "experiment_test.json"
    art.write_text(json.dumps(head_obj, indent=2) + "\n")
    run("git", "add", "-A")
    run("git", "commit", "-q", "-m", "base")
    art.write_text(json.dumps(cur_obj, indent=2) + "\n")

    sys.path.insert(0, str(REPO / "scripts"))
    import importlib

    rc = importlib.import_module("research_conductor")
    # `run_cmd` pins `cwd=PROJECT_ROOT`, so os.chdir cannot redirect it at the throwaway repo.
    # Point PROJECT_ROOT at tmp_path for the call instead of relaxing the real code, and also
    # chdir so the helper's relative `pathlib.Path(rel)` writes land in the same place.
    import os

    prev_root, prev_cwd = rc.PROJECT_ROOT, os.getcwd()
    rc.PROJECT_ROOT = tmp_path
    os.chdir(tmp_path)
    try:
        rc._restore_dropped_determinations()
    finally:
        rc.PROJECT_ROOT = prev_root
        os.chdir(prev_cwd)
    return json.loads(art.read_text())


def test_a_deleted_determination_is_restored(tmp_path) -> None:
    out = _restore_in(
        tmp_path,
        {"experiment": "t", "flagged_adversarial": True, "corrigendum_note": "why"},
        {"experiment": "t"},
    )
    assert out["flagged_adversarial"] is True
    assert out["corrigendum_note"] == "why"


def test_a_determination_downgraded_to_none_is_restored(tmp_path) -> None:
    """THE 2026-08-12 SHAPE. The key is present and nulled, so `k not in cur` was False and
    the original helper restored nothing on exactly the case it was written for."""
    out = _restore_in(
        tmp_path,
        {"experiment": "t", "flagged_adversarial": True, "corrigendum_note": "why"},
        {"experiment": "t", "flagged_adversarial": None, "corrigendum_note": None},
    )
    assert out["flagged_adversarial"] is True
    assert out["corrigendum_note"] == "why"


def test_a_determination_downgraded_to_false_is_restored(tmp_path) -> None:
    # False re-admits the artifact to headline aggregation just as None does.
    out = _restore_in(
        tmp_path,
        {"experiment": "t", "flagged_adversarial": True},
        {"experiment": "t", "flagged_adversarial": False},
    )
    assert out["flagged_adversarial"] is True


def test_a_deliberate_clear_with_a_cleared_note_is_LEFT_ALONE(tmp_path) -> None:
    """The sanctioned route must survive. determination_preservation_lint documents clearing
    as: set the value falsy AND add a `*_cleared_note` saying what was re-verified. Restoring
    over that would make an auditable decision impossible to express."""
    out = _restore_in(
        tmp_path,
        {"experiment": "t", "flagged_adversarial": True},
        {
            "experiment": "t",
            "flagged_adversarial": False,
            "flagged_adversarial_cleared_note": "re-verified: TAUTOLOGY was structural",
        },
    )
    assert out["flagged_adversarial"] is False
    assert "cleared_note" in " ".join(out)


def test_new_measurements_alongside_a_restored_determination_are_preserved(tmp_path) -> None:
    # Fail-forward: a re-run's fresh numbers must survive the repair.
    out = _restore_in(
        tmp_path,
        {"experiment": "t", "flagged_adversarial": True, "auroc": 0.5},
        {"experiment": "t", "flagged_adversarial": None, "auroc": 0.91, "new_field": 7},
    )
    assert out["flagged_adversarial"] is True
    assert out["auroc"] == 0.91
    assert out["new_field"] == 7


def test_an_artifact_with_nothing_missing_is_not_touched(tmp_path) -> None:
    obj = {"experiment": "t", "flagged_adversarial": True}
    out = _restore_in(tmp_path, obj, dict(obj))
    assert out == obj
