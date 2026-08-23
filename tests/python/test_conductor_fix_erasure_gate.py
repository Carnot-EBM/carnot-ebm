"""Tests for the test-fix erasure gate (REQ-CONDUCTOR-FIXGATE-1).

Origin: 2026-08-23 live specimen — the conductor's test-fixer, told "fix
the failing tests" and "do NOT modify scripts/research_conductor.py",
added pytest.mark.skipif to the failing (untracked) test file and
reverted the foreign block the tests covered. Green suite, work erased.

Every test runs against a throwaway git repo under tmp_path. No test
touches the real tree; no test invokes an agent.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_conductor as rc  # noqa: E402


def _git(repo: Path, *args) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
            "HOME": str(repo),
        },
    )


def _mk_repo(tmp_path: Path) -> Path:
    """A committed repo with one source file and one tracked test file,
    plus the shape of a mid-landing task: a dirty tracked file and an
    untracked test file (the live specimen's exact shape)."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "src.py").write_text("VALUE = 1\n")
    (repo / "tests" / "test_old.py").write_text("def test_old():\n    assert True\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    # task edits: dirty tracked source + untracked new test, no skips
    (repo / "src.py").write_text("VALUE = 2  # the task's in-flight work\n")
    (repo / "tests" / "test_new.py").write_text(
        "import src\n\ndef test_new():\n    assert src.VALUE == 2\n"
    )
    return repo


def test_live_specimen_detected_and_restored(tmp_path):
    """SCENARIO-CONDUCTOR-FIXGATE-1-LIVE-SPECIMEN: skipif added to the
    untracked test file + tracked work reverted -> both reported, both
    restored byte-identically."""
    repo = _mk_repo(tmp_path)
    snapshot = rc._snapshot_task_edits(cwd=repo)
    assert snapshot is not None
    assert snapshot["src.py"]["tracked"] is True
    assert snapshot["tests/test_new.py"]["tracked"] is False
    task_test_bytes = (repo / "tests" / "test_new.py").read_bytes()
    task_src_bytes = (repo / "src.py").read_bytes()

    # the fixer's move, verbatim shape from the incident:
    (repo / "tests" / "test_new.py").write_text(
        "import pytest\nimport src\n\n"
        'pytestmark = pytest.mark.skipif(True, reason="task forbids modifying that file")\n\n'
        "def test_new():\n    assert src.VALUE == 2\n"
    )
    _git(repo, "checkout", "--", "src.py")  # work reverted

    erasure = rc._detect_fix_erasure(snapshot, cwd=repo)
    assert erasure is not None
    assert any("test_new.py" in s for s in erasure["added_skips"])
    assert erasure["reverted"] == ["src.py"]

    restored = rc._restore_erased(
        snapshot, sorted(set(erasure["skip_files"]) | set(erasure["reverted"])), cwd=repo
    )
    assert len(restored) == 2
    assert (repo / "tests" / "test_new.py").read_bytes() == task_test_bytes
    assert (repo / "src.py").read_bytes() == task_src_bytes


def test_clean_fix_reports_nothing(tmp_path):
    """SCENARIO-CONDUCTOR-FIXGATE-1-CLEAN-FIX."""
    repo = _mk_repo(tmp_path)
    snapshot = rc._snapshot_task_edits(cwd=repo)
    (repo / "src.py").write_text("VALUE = 2\nHELPER = 3  # a real logic fix\n")
    erasure = rc._detect_fix_erasure(snapshot, cwd=repo)
    assert erasure == {"added_skips": [], "skip_files": set(), "reverted": []}


def test_tracked_test_skip_detected_from_diff(tmp_path):
    """SCENARIO-CONDUCTOR-FIXGATE-1-TRACKED-SKIP."""
    repo = _mk_repo(tmp_path)
    snapshot = rc._snapshot_task_edits(cwd=repo)
    (repo / "tests" / "test_old.py").write_text(
        "import unittest\n\n@unittest.skip('later')\ndef test_old():\n    assert True\n"
    )
    erasure = rc._detect_fix_erasure(snapshot, cwd=repo)
    assert any("test_old.py" in s for s in erasure["added_skips"])
    assert "tests/test_old.py" in erasure["skip_files"]
    rc._restore_erased(snapshot, ["tests/test_old.py"], cwd=repo)
    assert "unittest.skip" not in (repo / "tests" / "test_old.py").read_text()


def test_preexisting_skip_in_untracked_file_is_not_new(tmp_path):
    """A skip already present at snapshot time is standing debt, not a
    fixer move — the gate flags NEW erasure only."""
    repo = _mk_repo(tmp_path)
    (repo / "tests" / "test_new.py").write_text(
        "import pytest\npytestmark = pytest.mark.skipif(False, reason='preexisting')\n"
        "def test_new():\n    assert True\n"
    )
    snapshot = rc._snapshot_task_edits(cwd=repo)
    erasure = rc._detect_fix_erasure(snapshot, cwd=repo)
    assert erasure["added_skips"] == []


def test_fixer_created_skip_file_is_removed(tmp_path):
    """A skip-bearing test file the fixer created from nothing (absent
    from the snapshot) is deleted on restore."""
    repo = _mk_repo(tmp_path)
    snapshot = rc._snapshot_task_edits(cwd=repo)
    ghost = repo / "tests" / "test_ghost.py"
    ghost.write_text("import pytest\npytestmark = pytest.mark.skip\n")
    erasure = rc._detect_fix_erasure(snapshot, cwd=repo)
    assert any("test_ghost.py" in s for s in erasure["added_skips"])
    rc._restore_erased(snapshot, sorted(erasure["skip_files"]), cwd=repo)
    assert not ghost.exists()


def test_git_unavailable_returns_none(tmp_path):
    """Rule 5 fail direction: not-a-repo means the gate cannot audit, and
    None tells the caller to reject the fix."""
    bare = tmp_path / "not_a_repo"
    bare.mkdir()
    assert rc._snapshot_task_edits(cwd=bare) is None
    assert rc._detect_fix_erasure({}, cwd=bare) is None


def _code_only(source: str) -> str:
    lines = []
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append(code)
    return "\n".join(lines)


def test_gate_is_wired_into_the_fix_loop():
    """A check nothing calls is the bug class."""
    source = _code_only(inspect.getsource(rc.research_step))
    assert "_snapshot_task_edits()" in source
    assert "_detect_fix_erasure(pre_fix_snapshot)" in source
    assert "_restore_erased(pre_fix_snapshot" in source


def test_fix_prompt_forbids_erasure_repairs():
    """Rule 4: the fixer is TOLD, not just caught."""
    source = inspect.getsource(rc.research_step)
    assert "FORBIDDEN repairs" in source
    assert "pytest.mark.skip" in source
    assert "reverting a modified file" in source


def test_selfedit_revert_preserves_the_diff():
    """Rule 6: the self-edit revert rescues the diff and logs durably."""
    source = _code_only(inspect.getsource(rc.research_step))
    assert "SELFEDIT_RESCUE_DIR" in source
    assert "Conductor self-edit reverted" in source
