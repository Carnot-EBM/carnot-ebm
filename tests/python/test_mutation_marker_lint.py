"""A mutation proof's marker must never reach a commit.

REQ-OPS-MUTATION-PROOF-2 / SCENARIOs: marker-in-staged-python-refuses,
allowlisted-definition-passes, unreadable-refuses, non-python-ignored.

INCIDENT, 2026-08-25. Two hand-run mutation proofs collided in one working
tree and left

    python/carnot/agentic/arc_executable_world_model.py:6466: pass  # MUTATED M6

on the LIVE ARC scored path. That line is valid Python -- it parses, imports,
and clears every other hook in this repo -- while the conductor commits on its
own schedule with hooks skipped. Nothing in the config would have stopped it.

The sibling mutation-PROOF session is OPT-IN: an agent has to remember to wrap.
This hook does not care whether a session was used; it watches for the harm.

All writes go under tmp_path; no test touches tracked state.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import mutation_marker_lint as mml  # noqa: E402

_MARKER = "MUTATED"
_INCIDENT_LINE = "    pass  # " + _MARKER + " M6\n"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo with the lint pointed at it."""
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "seed.py").write_text("x = 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")
    monkeypatch.setattr(mml, "REPO", r)
    return r


def test_a_marker_in_staged_python_refuses_the_commit(repo, capsys):
    """THE INCIDENT LINE, staged. This is the whole point of the hook."""
    victim = repo / "victim.py"
    victim.write_text("def f():\n" + _INCIDENT_LINE)
    _git(repo, "add", "victim.py")
    assert mml.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSING THE COMMIT" in out
    assert "victim.py:2" in out, "the offending line must be named"


def test_the_marker_is_imported_from_the_module_that_defines_it():
    """One list, one home. A copy here is how a lint silently stops matching
    the convention it was written for."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tsm_for_marker_test", REPO / "scripts" / "test_suite_mutation_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("tsm_for_marker_test", module)
    spec.loader.exec_module(module)
    assert mml._marker() == module.MUTATION_MARKER


def test_a_clean_staged_python_file_passes(repo):
    (repo / "clean.py").write_text("def f():\n    return 1\n")
    _git(repo, "add", "clean.py")
    assert mml.main([]) == 0


def test_the_allowlist_is_exactly_the_files_that_must_define_the_token():
    """Four entries, each unable to do its job without the literal word.
    Everything else in the repo is scanned -- an allow-list, not a filter."""
    assert mml.ALLOWLIST == {
        "scripts/mutation_marker_lint.py",
        "scripts/test_suite_mutation_check.py",
        "tests/python/test_mutation_marker_lint.py",
        "tests/python/test_test_suite_mutation_check.py",
    }


def test_an_allowlisted_file_carrying_the_token_does_not_refuse(repo):
    """The guard's own source must be committable."""
    guard = repo / "scripts" / "test_suite_mutation_check.py"
    guard.write_text("MUTATION_MARKER = " + repr(_MARKER) + "\n")
    _git(repo, "add", "-A")
    assert mml.main([]) == 0


def test_the_real_repo_is_clean_under_this_lint():
    """The lint must not fire on the tracked tree it was just added to. If it
    does, it would refuse every commit -- the self-inflicted brick its sibling
    already committed once."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [REPO / n for n in tracked.stdout.split("\0") if n]
    assert paths, "expected tracked Python files"
    # worktree source: one git show per file would be thousands of subprocesses.
    assert mml.scan(paths, source="worktree") == []


def test_a_non_python_file_carrying_the_token_is_ignored(repo):
    """Prose that quotes the marker while documenting the incident is not a
    mutation. ops/changelog.md and the spec both do exactly this."""
    (repo / "changelog.md").write_text("The tree carried `" + _INCIDENT_LINE.strip() + "`.\n")
    _git(repo, "add", "changelog.md")
    assert mml.main([]) == 0


def test_the_index_is_read_not_the_working_tree(repo, capsys):
    """The question is what the COMMIT will contain. A marker staged and then
    removed from disk must still refuse; a marker only on disk must not."""
    victim = repo / "victim.py"
    victim.write_text("def f():\n" + _INCIDENT_LINE)
    _git(repo, "add", "victim.py")
    victim.write_text("def f():\n    return 1\n")  # cleaned on disk, still staged dirty
    assert mml.main([]) == 1
    assert "victim.py:2" in capsys.readouterr().out


def test_a_marker_only_in_the_working_tree_does_not_refuse(repo):
    """Unstaged work in flight is not a commit. Refusing here would block every
    commit during a legitimate in-progress proof."""
    (repo / "unstaged.py").write_text("def f():\n" + _INCIDENT_LINE)
    assert mml.main([]) == 0


def test_git_failure_refuses_rather_than_reporting_clean(repo, capsys, monkeypatch):
    """Fail closed. A guard that answers 'clean' when it could not look is the
    failure this repo keeps re-learning."""

    def boom(*_a, **_k):
        raise mml.LintError("git exploded")

    monkeypatch.setattr(mml, "staged_python", boom)
    assert mml.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSING THE COMMIT" in out and "could not run" in out


def test_an_undecodable_staged_file_refuses(repo, capsys):
    """A .py that will not decode cannot be scanned, so it cannot be cleared."""
    bad = repo / "bad.py"
    bad.write_bytes(b"x = 1\n\xff\xfe not utf-8\n")
    _git(repo, "add", "bad.py")
    assert mml.main([]) == 1
    assert "REFUSING THE COMMIT" in capsys.readouterr().out


def test_the_pattern_does_not_fire_on_a_longer_word(repo):
    """`\\bMUTATED` must not match PERMUTATED. Word-boundary blindness is a
    bug class this project has been bitten by (the 'meta' substring incident)."""
    (repo / "perm.py").write_text("PERMUTATED = 1\nunpermutated = 2\n")
    _git(repo, "add", "perm.py")
    assert mml.main([]) == 0


def test_the_pattern_still_fires_on_a_suffixed_marker(repo, capsys):
    """`# MUTATED_M6` is the same convention with an underscore. Requiring a
    word END too would let it through, so the boundary is word-START only."""
    (repo / "suffixed.py").write_text("x = 1  # " + _MARKER + "_M6\n")
    _git(repo, "add", "suffixed.py")
    assert mml.main([]) == 1
    assert "suffixed.py:1" in capsys.readouterr().out


def test_explicit_filenames_are_honoured(repo, capsys):
    """pre-commit passes filenames; the tool must scan those rather than
    re-deriving its own list."""
    victim = repo / "victim.py"
    victim.write_text("def f():\n" + _INCIDENT_LINE)
    _git(repo, "add", "victim.py")
    assert mml.main([str(victim)]) == 1
    assert "victim.py:2" in capsys.readouterr().out


def test_the_hook_is_wired_into_pre_commit():
    """A guard nothing calls is the bug class this hook exists to close. Its
    sibling --check-targets shipped with no caller for exactly this reason."""
    config = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "mutation-marker-lint" in config
    assert "scripts/mutation_marker_lint.py" in config
