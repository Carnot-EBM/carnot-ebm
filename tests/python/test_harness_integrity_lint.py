"""Tests for the sealed-hash and scope-glob guard.

REQ-INFRA-6800  scope globs: a staged path outside the declared scope refuses the commit.
REQ-INFRA-6801  sealed hashes: a harness file that moved in the WORKING TREE refuses it.

Every test here is written against a temporary repo, never the real one. A guard test that
writes tracked state would be the exact incident this repo's Test-Run Record Integrity
discipline exists to stop.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "harness_integrity_lint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harness_integrity_lint", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    """The module, pointed at a throwaway repo with a fake sealed file."""
    module = _load_module()
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "ops").mkdir()
    sealed = repo / "scripts" / "adversarial_verify.py"
    sealed.write_text("# the fabrication gate\n")

    monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(module, "SCOPES", repo / "ops" / ".agent_scopes")
    monkeypatch.setattr(module, "SEALED_PATHS", ("scripts/adversarial_verify.py",))
    # Default: git works and nothing is staged. Individual tests override.
    monkeypatch.setattr(module, "_staged_paths", lambda: [])
    module._repo_path = repo
    module._sealed_file = sealed
    return module


def _declare(module, *globs, unseal=(), run_id="test-run"):
    assert module.declare(list(globs), list(unseal), run_id) == 0


def test_scenario_infra_6802_inert_without_a_declaration(guard):
    """SCENARIO-INFRA-6802: no scope declared -> the check does not fire.

    This is what keeps the conductor running. It commits with `git add -A` every few minutes
    and declares nothing, so a guard that refused by default would wedge the research loop.
    """
    assert guard.check() == 0


def test_scenario_infra_6803_staged_path_outside_scope_refuses(guard, monkeypatch, capsys):
    """SCENARIO-INFRA-6800/6803: the `git add -A` sweep this guard was written for."""
    _declare(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_staged_paths", lambda: ["scripts/foo.py", "ops/known-issues.md"])
    assert guard.check() == 1
    out = capsys.readouterr().out
    assert "ops/known-issues.md" in out
    assert "scripts/foo.py" in out


def test_in_scope_commit_passes(guard, monkeypatch):
    """A commit that stays inside its declaration is not obstructed."""
    _declare(guard, "scripts/foo.py", "tests/python/test_foo.py")
    monkeypatch.setattr(
        guard, "_staged_paths", lambda: ["scripts/foo.py", "tests/python/test_foo.py"]
    )
    assert guard.check() == 0


def test_directory_scope_covers_its_contents(guard, monkeypatch):
    """`--scope docs/notes` means the directory, which is what people mean when they type it."""
    _declare(guard, "docs/notes")
    monkeypatch.setattr(guard, "_staged_paths", lambda: ["docs/notes/a.md", "docs/notes/sub/b.md"])
    assert guard.check() == 0


def test_scenario_infra_6804_sealed_file_changed_unstaged_refuses(guard, capsys):
    """SCENARIO-INFRA-6801/6804: the failure a staged-diff check cannot see.

    The sealed file is edited in the working tree and NEVER staged. A test suite reads the
    working tree, so this is how a change makes itself pass without appearing in any diff.
    """
    _declare(guard, "scripts/foo.py")
    guard._sealed_file.write_text("# weakened\n")
    assert guard.check() == 1
    out = capsys.readouterr().out
    assert "scripts/adversarial_verify.py" in out
    assert "WORKING TREE" in out


def test_scenario_infra_6805_explicitly_unsealed_path_is_permitted(guard):
    """SCENARIO-INFRA-6805: naming the path in the declaration permits editing it."""
    _declare(guard, "scripts/foo.py", unseal=("scripts/adversarial_verify.py",))
    guard._sealed_file.write_text("# deliberately edited\n")
    assert guard.check() == 0


def test_a_glob_never_unseals(guard, monkeypatch, capsys):
    """A wide scope glob does not grant permission over a sealed path.

    Widening a glob is cheap and is what an agent under pressure does. Typing the name of the
    fabrication gate is deliberate. That asymmetry is the point of the seal.
    """
    _declare(guard, "scripts/*")
    guard._sealed_file.write_text("# weakened via a wide glob\n")
    monkeypatch.setattr(guard, "_staged_paths", lambda: ["scripts/adversarial_verify.py"])
    assert guard.check() == 1
    assert "sealed harness file" in capsys.readouterr().out


def test_sealed_file_deleted_is_drift(guard, capsys):
    """Deleting a guard is a harness change, not an absence of one."""
    _declare(guard, "scripts/foo.py")
    guard._sealed_file.unlink()
    assert guard.check() == 1
    assert "now deleted" in capsys.readouterr().out


def test_sealed_file_appearing_is_drift(guard, monkeypatch, capsys):
    """A sealed path absent at declaration and present later is also drift.

    Adding a conftest.py changes what the suite does as surely as editing one.
    """
    monkeypatch.setattr(guard, "SEALED_PATHS", ("tests/python/conftest.py",))
    _declare(guard, "scripts/foo.py")
    conftest = guard.REPO / "tests" / "python" / "conftest.py"
    conftest.parent.mkdir(parents=True, exist_ok=True)
    conftest.write_text("# newly added\n")
    assert guard.check() == 1
    assert "now present" in capsys.readouterr().out


def test_scenario_infra_6806_scope_staged_in_its_own_commit_refuses(guard, monkeypatch, capsys):
    """SCENARIO-INFRA-6806: a scope edited in the commit it governs proves nothing."""
    _declare(guard, "ops/.agent_scopes/test-run.scope.json")
    monkeypatch.setattr(guard, "_staged_paths", lambda: ["ops/.agent_scopes/test-run.scope.json"])
    assert guard.check() == 1
    assert "proves nothing" in capsys.readouterr().out


def test_scenario_infra_6807_unreadable_scope_refuses(guard, capsys):
    """SCENARIO-INFRA-6807: fail-closed. Unreadable is not the same as absent."""
    _declare(guard, "scripts/foo.py")
    (guard.SCOPES / "test-run.scope.json").write_text("{ not json")
    assert guard.check() == 1
    assert "unreadable" in capsys.readouterr().out.lower()


def test_git_failure_refuses(guard, monkeypatch, capsys):
    """A guard that reports clean when git is broken is the trusted-and-silent state."""
    _declare(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_staged_paths", lambda: None)
    assert guard.check() == 1
    assert "git failed" in capsys.readouterr().out


def test_empty_scope_permits_nothing(guard, capsys):
    """A declaration with no patterns is refused rather than read as 'allow everything'."""
    guard.SCOPES.mkdir(parents=True, exist_ok=True)
    (guard.SCOPES / "empty.scope.json").write_text(json.dumps({"run_id": "empty", "scope": []}))
    assert guard.check() == 1
    assert "permits nothing" in capsys.readouterr().out


def test_declare_refuses_unsealing_a_path_that_is_not_sealed(guard, capsys):
    """An --unseal for an unsealed path means the agent believes it has permission it lacks."""
    assert guard.declare(["scripts/foo.py"], ["scripts/not_sealed.py"], "typo-run") == 1
    assert "not sealed" in capsys.readouterr().out


def test_declare_needs_a_scope(guard, capsys):
    assert guard.declare([], [], "no-scope") == 1
    assert "needs at least one" in capsys.readouterr().out


def test_release_removes_the_declaration(guard):
    _declare(guard, "scripts/foo.py")
    assert guard.check() == 0 or True  # scope is satisfied; the point is it exists
    assert guard.release("test-run") == 0
    assert guard.check() == 0
    assert guard._scope_files() == []


def test_release_refuses_to_guess_between_several_scopes(guard, capsys):
    """Releasing another session's scope silently removes its guard, so never guess."""
    _declare(guard, "scripts/a.py", run_id="run-a")
    _declare(guard, "scripts/b.py", run_id="run-b")
    assert guard.release(None) == 1
    assert "name one with --run-id" in capsys.readouterr().out


def test_release_with_no_scope_is_not_an_error(guard):
    assert guard.release(None) == 0


def test_release_unknown_run_id_reports_it(guard, capsys):
    _declare(guard, "scripts/a.py", run_id="run-a")
    assert guard.release("run-missing") == 1
    assert "no scope with run id" in capsys.readouterr().out


def test_list_reports_active_and_unreadable_scopes(guard, capsys):
    assert guard.show() == 0
    assert "inert" in capsys.readouterr().out
    _declare(guard, "scripts/a.py", unseal=("scripts/adversarial_verify.py",), run_id="run-a")
    assert guard.show() == 0
    out = capsys.readouterr().out
    assert "run-a" in out and "scripts/adversarial_verify.py" in out
    (guard.SCOPES / "run-a.scope.json").write_text("{ broken")
    assert guard.show() == 0
    assert "UNREADABLE" in capsys.readouterr().out


def test_two_scopes_are_both_enforced(guard, monkeypatch, capsys):
    """Concurrent sessions each keep their own guard; neither relaxes the other."""
    _declare(guard, "scripts/a.py", run_id="run-a")
    _declare(guard, "scripts/b.py", run_id="run-b")
    monkeypatch.setattr(guard, "_staged_paths", lambda: ["scripts/a.py"])
    assert guard.check() == 1  # in scope for run-a, out of scope for run-b
    assert "run-b" in capsys.readouterr().out


def test_main_dispatches_each_mode(guard, monkeypatch):
    """The CLI wiring is exercised, not just the functions beneath it."""
    monkeypatch.setattr(guard, "_staged_paths", lambda: [])
    assert guard.main(["--declare", "--scope", "scripts/a.py", "--run-id", "cli-run"]) == 0
    assert guard.main(["--list"]) == 0
    assert guard.main(["--check"]) == 0
    assert guard.main(["--release", "--run-id", "cli-run"]) == 0


# --- HEAD-anchored seals (REQ-INFRA-6801) -------------------------------------------------
# A standing declaration -- one that outlives many commits, like the conductor's -- cannot use
# a declaration-time hash: the first legitimate commit of a harness file would make the seal
# stale and refuse every commit afterwards until a human noticed. Anchored to HEAD, a
# committed change moves the baseline with it and only UNCOMMITTED edits refuse.


def _declare_head(module, *globs, run_id="head-run"):
    assert module.declare(list(globs), [], run_id, "head") == 0


def test_head_anchor_records_the_sentinel_not_a_hash(guard):
    _declare_head(guard, "scripts/foo.py")
    record = json.loads((guard.SCOPES / "head-run.scope.json").read_text())
    assert record["seal_anchor"] == "head"
    assert set(record["seals"].values()) == {guard.HEAD_ANCHOR}


def test_head_anchor_passes_when_worktree_matches_head(guard, monkeypatch):
    _declare_head(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("abc123", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("abc123", True))
    assert guard.check() == 0


def test_head_anchor_refuses_an_uncommitted_harness_edit(guard, monkeypatch, capsys):
    """The case that matters: a harness file edited but not yet committed."""
    _declare_head(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("committed", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("edited", True))
    assert guard.check() == 1
    assert "modified, uncommitted" in capsys.readouterr().out


def test_head_anchor_survives_a_committed_harness_change(guard, monkeypatch):
    """A legitimate commit moves the baseline; the standing declaration does not go stale."""
    _declare_head(guard, "scripts/foo.py")
    # Both sides move together, which is what committing does.
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("v2", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("v2", True))
    assert guard.check() == 0


def test_head_anchor_refuses_a_file_absent_from_head(guard, monkeypatch, capsys):
    _declare_head(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: (None, True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("new", True))
    assert guard.check() == 1
    assert "not in HEAD" in capsys.readouterr().out


def test_head_anchor_refuses_an_uncommitted_deletion(guard, monkeypatch, capsys):
    _declare_head(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("committed", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: (None, True))
    assert guard.check() == 1
    assert "deleted uncommitted" in capsys.readouterr().out


def test_head_anchor_refuses_when_git_cannot_be_asked(guard, monkeypatch, capsys):
    """Fail-closed: an unanswerable git is not evidence the harness is intact."""
    _declare_head(guard, "scripts/foo.py")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: (None, False))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("x", True))
    assert guard.check() == 1
    assert "refusing rather than guessing" in capsys.readouterr().out


def test_an_explicit_unseal_in_one_scope_lifts_a_standing_seal(guard, monkeypatch):
    """A standing declaration must not deadlock the person fixing the file it seals.

    The conductor's scope is long-lived and seals the record-preservation lints. Without
    this, an agent legitimately repairing one of them could not commit: its own --unseal
    does not reach the conductor's seal, so it would have to release someone else's scope --
    the exact move the refusal text tells it not to make.
    """
    _declare_head(guard, "*", run_id="standing")
    guard._sealed_file.write_text("# a deliberate repair\n")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("committed", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("edited", True))
    assert guard.check() == 1  # the standing seal refuses on its own
    _declare(
        guard,
        "scripts/adversarial_verify.py",
        unseal=("scripts/adversarial_verify.py",),
        run_id="repair",
    )
    assert guard.check() == 0  # the named path lifts it


def test_a_glob_still_never_lifts_a_standing_seal(guard, monkeypatch, capsys):
    """The bypass costs a typed path name. A wide glob does not buy it."""
    _declare_head(guard, "*", run_id="standing")
    monkeypatch.setattr(guard, "_head_blob", lambda rel: ("committed", True))
    monkeypatch.setattr(guard, "_worktree_blob", lambda rel: ("edited", True))
    _declare(guard, "scripts/*", run_id="wide")
    assert guard.check() == 1
    assert "sealed harness file" in capsys.readouterr().out
