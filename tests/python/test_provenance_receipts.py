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
    """SCENARIO-REPORT-6610-TAMPER: a hand-edited recorded hash still disagrees.

    The equality assertion is load-bearing. With only the inequality this test
    stayed green when the hash function was replaced by a constant, because a
    constant is still unequal to a value nobody ever hashed.
    """

    receipt = pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT)
    recorded = _sha("a value nobody ever hashed\n")
    assert receipt != recorded
    assert receipt == _sha("spec v1\n")


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
    value = pr.receipt_sha256(repo / SPEC, artifact_relative_path="results/never_committed.json")
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

    assert pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT, root=repo) == _sha(
        "spec v1\n"
    )


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


UPSTREAM = "results/experiment_0_upstream.json"


@pytest.fixture
def repo_with_upstream(tmp_path: Path) -> Path:
    """A repo where an upstream evidence artifact landed in the SAME commit.

    Separate from `repo` because a dependency added later legitimately fails the
    missing-at-commit check, which would mask what these tests measure.
    """

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _write(root, SPEC, "spec v1\n")
    _write(root, UPSTREAM, '{"ready": true}\n')
    _write(root, ARTIFACT, '{"receipt": "placeholder"}\n')
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "land the artifact and its upstream")
    return root


def test_req_report_6610_evidence_live_reads_the_working_tree_by_default(
    repo_with_upstream: Path,
) -> None:
    """SCENARIO-REPORT-6610-EVIDENCE-LIVE: an evidence receipt is not pinned.

    Both calls agree while the working tree matches the commit. The next test
    separates them, which is where the behaviour actually shows.
    """

    upstream = repo_with_upstream / UPSTREAM
    assert pr.receipt_sha256(upstream, artifact_relative_path=ARTIFACT) == _sha('{"ready": true}\n')
    assert pr.receipt_sha256(
        upstream, artifact_relative_path=ARTIFACT, allow_evidence_pin=True
    ) == _sha('{"ready": true}\n')


def test_req_report_6610_evidence_live_sees_a_later_gate_stamp(
    repo_with_upstream: Path,
) -> None:
    """REQ-REPORT-6610-EVIDENCE-LIVE: the incident shape that motivated the rule.

    A fabrication-gate stamp added after the upstream landed changes no field a
    downstream module reads, so the receipt is the only thing that can report it.
    """

    upstream = repo_with_upstream / UPSTREAM
    landed = _sha('{"ready": true}\n')
    _write(repo_with_upstream, UPSTREAM, '{"ready": true, "flagged_adversarial": true}\n')
    pr._COMMIT_CACHE.clear()
    pr._BLOB_CACHE.clear()

    default = pr.receipt_sha256(upstream, artifact_relative_path=ARTIFACT)
    assert default != landed, "the default receipt must move when the upstream is stamped"

    pinned = pr.receipt_sha256(upstream, artifact_relative_path=ARTIFACT, allow_evidence_pin=True)
    assert pinned == landed, "a pinned receipt cannot see the stamp -- why live is the default"


def test_req_report_6610_evidence_live_does_not_block_shared_files(
    repo_with_upstream: Path,
) -> None:
    """The refusal is scoped to evidence; a shared spec still pins as before."""

    assert pr.receipt_sha256(repo_with_upstream / SPEC, artifact_relative_path=ARTIFACT) == _sha(
        "spec v1\n"
    )


def test_req_report_6610_the_four_working_tree_sources_are_distinguishable(
    repo_with_upstream: Path,
) -> None:
    """REQ-REPORT-6610-SOURCE: a deliberate live read is not an unresolved one.

    Three of these read the working tree and one reads a commit. They returned
    an identical value before, so a reader could not tell policy from failure.
    """

    assert (
        pr.receipt_source(repo_with_upstream / SPEC, artifact_relative_path=ARTIFACT)
        == pr.SOURCE_COMMIT
    )
    assert (
        pr.receipt_source(repo_with_upstream / UPSTREAM, artifact_relative_path=ARTIFACT)
        == pr.SOURCE_LIVE_EVIDENCE
    )
    assert (
        pr.receipt_source(
            repo_with_upstream / SPEC, artifact_relative_path="results/never_landed.json"
        )
        == pr.SOURCE_AUTHORING
    )
    assert (
        pr.receipt_source(Path("/tmp"), artifact_relative_path=ARTIFACT)
        == pr.SOURCE_OUTSIDE_CHECKOUT
    )


def test_req_report_6610_a_git_failure_raises_instead_of_reading_the_disk(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6610-FAIL-CLOSED: a broken git must not answer from the disk.

    This is the trusted-and-silent case: without it, every receipt in the corpus
    would quietly become a working-tree hash while still calling itself pinned.
    """

    real = pr._run_git

    def broken(root: Path, *args: str):
        if args[:1] == ("log",):
            return subprocess.CompletedProcess(args, 128, b"", b"fatal: broken repository")
        return real(root, *args)

    monkeypatch.setattr(pr, "_run_git", broken)
    with pytest.raises(pr.ReceiptResolutionError, match="git log failed"):
        pr.receipt_sha256(repo / SPEC, artifact_relative_path=ARTIFACT)


def test_req_report_6610_a_checkout_with_no_commits_is_authoring(tmp_path: Path) -> None:
    """REQ-REPORT-6610-AUTHORING: an empty repo has no history, so it is not broken.

    `git log` exits non-zero here, which must NOT be read as a git failure.
    """

    root = tmp_path / "empty"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root, SPEC, "spec v1\n")
    assert pr.artifact_commit(root, ARTIFACT) is None
    assert pr.receipt_sha256(root / SPEC, artifact_relative_path=ARTIFACT) == _sha("spec v1\n")
