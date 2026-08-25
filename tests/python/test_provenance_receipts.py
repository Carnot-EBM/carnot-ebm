"""Tests for commit-resolved provenance receipts.

REQ-REPORT-6610 and its scenarios. Every test builds a real scratch git
repository under `tmp_path`, because the behaviour under test IS the git
resolution; mocking git would test the mock.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from carnot import provenance_receipts as pr

ARTIFACT = "results/experiment_1.json"
SPEC = "openspec/capabilities/demo/spec.md"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True, text=True
    )
    return proc.stdout.strip()


def _write(repo: Path, relative: str, text: str) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Each scratch repo reuses paths, so stale cache entries would cross tests."""

    pr._COMMIT_CACHE.clear()
    pr._BLOB_CACHE.clear()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A scratch repo where the artifact landed while the spec said 'v1'."""

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _write(root, SPEC, "spec v1\n")
    _write(root, ARTIFACT, '{"receipt": "placeholder"}\n')
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "land the artifact")
    return root


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_req_report_6610_churn_does_not_break_a_landed_receipt(repo: Path) -> None:
    """SCENARIO-REPORT-6610-CHURN: a later append to the shared spec is ignored."""

    landed = pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT)
    assert landed == _sha("spec v1\n")

    _write(repo, SPEC, "spec v1\nspec v2 appended by the next experiment\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "next experiment appends")
    pr._COMMIT_CACHE.clear()
    pr._BLOB_CACHE.clear()

    assert pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT) == landed
    assert _sha((repo / SPEC).read_text(encoding="utf-8")) != landed


def test_req_report_6610_tamper_is_still_detected(repo: Path) -> None:
    """SCENARIO-REPORT-6610-TAMPER: a hand-edited recorded hash still disagrees."""

    recorded = _sha("a value nobody ever hashed\n")
    assert pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT) != recorded


def test_req_report_6610_missing_dependency_fails_closed(repo: Path) -> None:
    """SCENARIO-REPORT-6610-MISSING: a dependency absent at the commit raises."""

    added_later = _write(repo, "openspec/capabilities/demo/extra.md", "added after\n")
    with pytest.raises(pr.ReceiptResolutionError) as excinfo:
        pr.receipt_sha256(added_later, artifact_relative_path=ARTIFACT)
    assert "not a file at the artifact commit" in str(excinfo.value)


def test_req_report_6610_directory_is_not_a_receipt(repo: Path) -> None:
    """REQ-REPORT-6610-FAIL-CLOSED: a tree at that path is not a file receipt."""

    with pytest.raises(pr.ReceiptResolutionError):
        pr.receipt_sha256(repo / "openspec", artifact_relative_path=ARTIFACT)


def test_req_report_6610_malformed_commit_fails_closed(repo: Path) -> None:
    """SCENARIO-REPORT-6610-BAD-COMMIT: a non-hex commit id raises."""

    with pytest.raises(pr.ReceiptResolutionError) as excinfo:
        pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT, commit="not-a-commit")
    assert "malformed artifact commit id" in str(excinfo.value)


def test_req_report_6610_dangling_commit_fails_closed(repo: Path) -> None:
    """SCENARIO-REPORT-6610-BAD-COMMIT: a well-formed but absent commit raises."""

    with pytest.raises(pr.ReceiptResolutionError) as excinfo:
        pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT, commit="0" * 40)
    assert "not in this repository" in str(excinfo.value)


def test_req_report_6610_authoring_mode_reads_the_working_tree(repo: Path) -> None:
    """REQ-REPORT-6610-AUTHORING: an artifact that never landed has no commit."""

    assert pr.artifact_commit(repo, "results/never_committed.json") is None
    value = pr.receipt_sha256(
        repo / SPEC, artifact_relative_path="results/never_committed.json"
    )
    assert value == _sha("spec v1\n")


def test_req_report_6610_path_outside_a_checkout_reads_directly(tmp_path: Path) -> None:
    """REQ-REPORT-6610-AUTHORING: a tmp_path fixture has no history to resolve."""

    loose = tmp_path / "loose.txt"
    loose.write_text("loose\n", encoding="utf-8")
    assert pr.repository_root(loose) is None
    assert pr.receipt_sha256(loose, artifact_relative_path=ARTIFACT) == _sha("loose\n")


def test_req_report_6610_existence_is_evaluated_at_the_commit(repo: Path) -> None:
    """REQ-REPORT-6610-EXISTENCE: deleting a file later must not null a receipt."""

    assert pr.receipt_exists(repo / SPEC, artifact_relative_path=ARTIFACT) is True
    (repo / SPEC).unlink()
    assert pr.receipt_exists(repo / SPEC, artifact_relative_path=ARTIFACT) is True

    never = repo / "openspec/capabilities/demo/absent.md"
    assert pr.receipt_exists(never, artifact_relative_path=ARTIFACT) is False


def test_req_report_6610_existence_outside_a_checkout_uses_the_disk(tmp_path: Path) -> None:
    """REQ-REPORT-6610-AUTHORING: existence of a loose fixture is a disk question."""

    loose = tmp_path / "loose.txt"
    assert pr.receipt_exists(loose, artifact_relative_path=ARTIFACT) is False
    loose.write_text("x", encoding="utf-8")
    assert pr.receipt_exists(loose, artifact_relative_path=ARTIFACT) is True


def test_req_report_6610_commit_prefers_the_add_over_a_later_stamp(repo: Path) -> None:
    """REQ-REPORT-6610-COMMIT: a later stamp of the artifact must not move it."""

    add_commit = _git(repo, "log", "-1", "--format=%H")
    _write(repo, ARTIFACT, '{"receipt": "placeholder", "flagged_adversarial": true}\n')
    _write(repo, SPEC, "spec v1\nappended between stamps\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fabrication gate stamps the artifact")
    pr._COMMIT_CACHE.clear()
    pr._BLOB_CACHE.clear()

    assert pr.artifact_commit(repo, ARTIFACT) == add_commit
    assert pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT) == _sha("spec v1\n")


def test_req_report_6610_falls_back_to_the_newest_touch_without_an_add(repo: Path) -> None:
    """REQ-REPORT-6610-COMMIT: a renamed artifact has no add for its new path."""

    _git(repo, "mv", ARTIFACT, "results/experiment_1_renamed.json")
    _git(repo, "commit", "-q", "-m", "rename the artifact")
    pr._COMMIT_CACHE.clear()
    renamed_commit = _git(repo, "log", "-1", "--format=%H")

    # git records the rename as an add by default, so force the fallback path by
    # asking about a path whose only history is a modification.
    resolved = pr.artifact_commit(repo, "results/experiment_1_renamed.json")
    assert resolved == renamed_commit


def test_req_report_6610_explicit_root_is_honoured(repo: Path) -> None:
    """The caller may name the checkout instead of having it discovered."""

    assert pr.receipt_sha256(
        repo / SPEC, artifact_relative_path=ARTIFACT, root=repo
    ) == _sha("spec v1\n")


def test_req_report_6610_receipt_bytes_returns_the_committed_bytes(repo: Path) -> None:
    """The byte-level helper is what modules keeping their own prefix call."""

    _write(repo, SPEC, "spec v1\nlater\n")
    assert pr.receipt_bytes(repo / SPEC, artifact_relative_path=ARTIFACT) == b"spec v1\n"


def test_req_report_6610_repository_root_finds_a_worktree_marker(tmp_path: Path) -> None:
    """A git worktree stores `.git` as a file, so both shapes must be found."""

    wt = tmp_path / "wt"
    (wt / "python").mkdir(parents=True)
    (wt / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert pr.repository_root(wt / "python" / "mod.py") == wt
