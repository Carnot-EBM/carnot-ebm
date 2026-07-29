"""Tests for the operator-curated-document write guard.

Spec coverage: CLAUDE.md "Public Documentation Discipline".

The guard is the thing that stops the test suite from rewriting the project's front door, so
these tests care about two properties in particular, both of which are ways a guard can look
installed and not actually protect anything:

  1. It must fire on a write to a protected path INSIDE the repository -- including through the
     symlink alias the repo is also reachable by, and including the atomic write-then-rename
     pattern that never calls `open()` on the destination at all.
  2. It must NOT fire on reads, and must NOT fire on writes to a `tmp_path` copy. A guard that
     blocks legitimate sandboxed work gets disabled, which is worse than no guard.
"""

# Test traces to REQ-ARC-WMTE-6043, SCENARIO-ARC-WMTE-6043-OPERATOR-CURATED-WRITE-IS-REFUSED.

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from carnot.testing import operator_curated_doc_guard as guard


REPO_ROOT = Path(__file__).resolve().parents[2]


def _lint_module():
    """Load `scripts/operator_curated_docs_lint.py` by path, without touching sys.path."""
    path = REPO_ROOT / "scripts" / "operator_curated_docs_lint.py"
    spec = importlib.util.spec_from_file_location("_operator_curated_docs_lint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_list_matches_the_commit_time_lint_exactly():
    """The guard duplicates the lint's path list; this stops the copies drifting apart.

    The duplication is deliberate (importing from `scripts/` inside `python/carnot/` would need
    a sys.path hack, which is itself a defect in this tree). Duplication is only safe while
    something mechanically holds the two lists equal -- that is this test.
    """
    assert guard.OPERATOR_CURATED_PATHS == _lint_module().OPERATOR_CURATED_PATHS


def test_readme_is_actually_protected_and_the_guard_is_installed():
    """Sanity: the hook is live in this session and README.md is in its watch set."""
    assert guard.is_installed(), "conftest.pytest_configure should have installed the guard"
    assert "README.md" in guard.OPERATOR_CURATED_PATHS
    assert guard.repo_root() == REPO_ROOT


def test_write_to_real_readme_is_refused():
    """The exact incident: a CWD-relative write to README.md from the repo root."""
    target = REPO_ROOT / "README.md"
    before = target.read_bytes()

    with pytest.raises(guard.OperatorCuratedDocWriteError, match="README.md"):
        target.write_text("clobbered")

    assert target.read_bytes() == before, "README.md must be byte-identical after a refused write"
    assert any(v["path"] == "README.md" for v in guard.recorded_violations())
    guard.clear_violations()


def test_write_through_a_symlink_alias_is_also_refused(tmp_path):
    """A symlinked route to the repo must not be an escape hatch.

    This checkout is reachable both as `.../ianblenke/carnot` and as the symlink
    `.../Carnot-EBM/carnot-ebm` (the venv lives under the alias, so real processes do run with
    it as their prefix). Two strings, one inode. If the guard compared unresolved strings, a
    write arriving by the alias would miss the watch list and the guard would appear to work
    while silently protecting nothing.

    The test builds its OWN symlink rather than depending on the machine-specific alias, so it
    proves the resolution property everywhere instead of only on the operator's box -- and so it
    never needs a skip, which CLAUDE.md forbids.
    """
    alias = tmp_path / "alias"
    alias.symlink_to(REPO_ROOT, target_is_directory=True)
    assert (alias / "README.md").resolve() == REPO_ROOT / "README.md"

    before = (REPO_ROOT / "README.md").read_bytes()

    with pytest.raises(guard.OperatorCuratedDocWriteError):
        (alias / "README.md").write_text("clobbered via alias")

    assert (REPO_ROOT / "README.md").read_bytes() == before
    guard.clear_violations()


def test_rename_onto_a_protected_path_is_refused(tmp_path):
    """The atomic-write pattern never opens the destination, so it needs its own event."""
    staged = tmp_path / "staged.md"
    staged.write_text("replacement")
    before = (REPO_ROOT / "README.md").read_bytes()

    with pytest.raises(guard.OperatorCuratedDocWriteError):
        os.replace(staged, REPO_ROOT / "README.md")

    assert (REPO_ROOT / "README.md").read_bytes() == before
    guard.clear_violations()


def test_reading_a_protected_document_is_allowed():
    """Reads must stay free -- tests/python/test_docs.py asserts on README's contents."""
    assert (REPO_ROOT / "README.md").read_text()
    assert guard.recorded_violations() == []


def test_writing_a_tmp_path_copy_is_allowed(tmp_path):
    """The sandboxed pattern that test_experiment_209_cleanup.py uses must keep working."""
    sandbox = tmp_path / "repo"
    (sandbox / "docs" / "blog").mkdir(parents=True)
    (sandbox / "README.md").write_text("# a copy, not the real one")
    (sandbox / "docs" / "index.html").write_text("<html></html>")
    (sandbox / "docs" / "blog" / "post.html").write_text("<html></html>")

    assert guard.recorded_violations() == []


def test_rmtree_of_a_sandbox_containing_a_readme_is_not_a_violation(tmp_path):
    """Regression test for the guard's own first false-positive.

    `shutil.rmtree` walks with `_rmtree_safe_fd`, which calls `os.unlink(entry.name,
    dir_fd=topfd)`. The `os.remove` audit event therefore carries the BARE NAME `"README.md"`,
    with nothing to say it is relative to a file descriptor pointing at a sandbox. The first
    version of this guard resolved that against the working directory -- the repo root -- and
    refused eight correctly-sandboxed tests, `test_experiment_209_cleanup.py` among them, purely
    because pytest was cleaning up a `tmp_path` that contained a `README.md` copy.

    This is the shape of bug CLAUDE.md's "QA-Layer Authenticity Discipline" is about: a check on
    the *record's* integrity that is itself wrong, and whose wrongness quarantines honest work.
    """
    import shutil

    sandbox = tmp_path / "sandbox"
    (sandbox / "docs").mkdir(parents=True)
    (sandbox / "README.md").write_text("a copy")
    (sandbox / "docs" / "index.html").write_text("<html></html>")

    shutil.rmtree(sandbox)  # must not raise

    assert not sandbox.exists()
    assert guard.recorded_violations() == []


def test_relative_paths_are_only_cwd_resolved_for_open_events():
    """The `allow_relative` split is load-bearing in BOTH directions; pin it.

    True  -- the incident itself was a relative write from the repo root.
    False -- an fd-relative `os.remove` carries a bare name that must NOT be CWD-resolved.
    """
    assert guard._violation_for("README.md", allow_relative=True) is not None
    assert guard._violation_for("README.md", allow_relative=False) is None
    # An absolute path is caught regardless of the flag.
    absolute = str(REPO_ROOT / "README.md")
    assert guard._violation_for(absolute, allow_relative=False) is not None


def test_blog_glob_entries_are_matched_inside_the_repo():
    """`docs/blog/*.html` is a glob, so it takes a different code path from the literals."""
    target = REPO_ROOT / "docs" / "blog" / "_guard_probe.html"
    with pytest.raises(guard.OperatorCuratedDocWriteError):
        target.write_text("should never be created")
    assert not target.exists()
    guard.clear_violations()


@pytest.mark.parametrize(
    ("mode", "flags", "expected"),
    [
        ("r", None, False),
        ("rb", None, False),
        ("w", None, True),
        ("a", None, True),
        ("x", None, True),
        ("r+", None, True),
        (None, os.O_RDONLY, False),
        (None, os.O_WRONLY | os.O_CREAT, True),
        (None, os.O_RDWR, True),
        (None, None, True),  # unrecognised form fails safe
    ],
)
def test_write_intent_classification(mode, flags, expected):
    """Both `open` audit-event shapes must be understood, and the unknown case must fail safe."""
    assert guard._is_write_intent(mode, flags) is expected
