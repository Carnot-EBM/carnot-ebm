"""REQ-INFRA-7087: a conductor-side hard file-size gate (ops/known-issues.md 2026-09-18).

INCIDENT. A 17GB cached GGUF model-weight blob was committed four separate times under a
moving per-experiment-ID `results/raw/experiment_NNNN/task_owned_model_cache/` path, plus
~9GB and ~30 other oversized files a static `.gitignore` list could not anticipate. GitHub
rejected the push outright once the aggregate pack exceeded its 2GB cap and per-file caps.
The fix required a full history rewrite (`git filter-repo --strip-blobs-bigger-than 90M`).
That rewrite closed the incident that already happened; it did nothing to stop the next one.

This gate is the size-based backstop the incident's own writeup queued: refuse to STAGE any
file over the threshold, regardless of path, at conductor checkpoint time -- before it ever
reaches a commit. It mirrors the existing mutation-proof exclusion in shape (a pure decision
rule, tested without a git fixture, plus call-site tests that bite `_stage_all_except_claimed`
directly) -- see test_conductor_mutation_proof_freeze.py for the precedent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import research_conductor as rc  # noqa: E402


# ---------------------------------------------------------------- the pure rule


def test_a_file_over_the_threshold_is_flagged() -> None:
    assert rc.oversized_staged_files(["a.bin"], {"a.bin": 100}, threshold_bytes=50) == ["a.bin"]


def test_a_file_at_or_under_the_threshold_is_kept() -> None:
    assert rc.oversized_staged_files(["a.py"], {"a.py": 50}, threshold_bytes=50) == []
    assert rc.oversized_staged_files(["a.py"], {"a.py": 10}, threshold_bytes=50) == []


def test_a_path_with_no_size_entry_is_never_oversized() -> None:
    """A staged deletion has nothing left to stat; absence must not read as infinite size."""
    assert rc.oversized_staged_files(["deleted.bin"], {}, threshold_bytes=50) == []


def test_only_the_oversized_subset_is_returned() -> None:
    staged = ["small.py", "huge.gguf", "medium.json"]
    sizes = {"small.py": 10, "huge.gguf": 999, "medium.json": 40}
    assert rc.oversized_staged_files(staged, sizes, threshold_bytes=50) == ["huge.gguf"]


def test_the_default_threshold_matches_the_incident_writeup() -> None:
    """ops/known-issues.md 2026-09-18 fixed the history rewrite at 90M; the live gate must
    match, or a file the rewrite would have stripped could still land in a future commit."""
    assert rc.OVERSIZED_FILE_THRESHOLD_BYTES == 90 * 1024 * 1024


# ------------------------------------------------------------- the call sites


def test_an_oversized_staged_file_is_actually_unstaged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bites the call site: the rule must reach `git restore --staged`, not merely exist."""
    (tmp_path / "huge.bin").write_bytes(b"x" * 100)
    (tmp_path / "small.py").write_bytes(b"x" * 10)
    calls: list[list[str]] = []

    def fake(cmd, *a, **k):  # noqa: ANN001, ANN202
        calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            return 0, "huge.bin\nsmall.py\n", ""
        return 0, "", ""

    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    rc._unstage_oversized_files(threshold_bytes=50)
    assert ["git", "restore", "--staged", "--", "huge.bin"] in calls


def test_nothing_is_unstaged_when_everything_is_under_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "small.py").write_bytes(b"x" * 10)
    calls: list[list[str]] = []

    def fake(cmd, *a, **k):  # noqa: ANN001, ANN202
        calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            return 0, "small.py\n", ""
        return 0, "", ""

    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    rc._unstage_oversized_files(threshold_bytes=50)
    assert not any("restore" in c for c in calls)


def test_a_staged_deletion_does_not_crash_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file no longer exists on disk (it was staged as a deletion) -- stat() raises, and
    the gate must skip it rather than propagate."""
    calls: list[list[str]] = []

    def fake(cmd, *a, **k):  # noqa: ANN001, ANN202
        calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            return 0, "gone.bin\n", ""
        return 0, "", ""

    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    rc._unstage_oversized_files(threshold_bytes=50)  # must not raise
    assert not any("restore" in c for c in calls)


def test_the_gate_fails_open_on_an_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAIL-OPEN per the docstring: a broken gate must not block the checkpoint commit."""

    def boom(cmd, *a, **k):  # noqa: ANN001, ANN202
        raise RuntimeError("git is unavailable")

    monkeypatch.setattr(rc, "run_cmd", boom)
    rc._unstage_oversized_files()  # must not raise


def test_the_staging_path_invokes_the_oversized_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bites the call site inside `_stage_all_except_claimed`: a rule that is correct and never
    invoked proves nothing. Must run BEFORE the scope-narrowing early-return, same reasoning as
    the mutation-proof exclusion it sits next to."""
    seen: list[str] = []
    monkeypatch.setattr(rc, "_restore_dropped_determinations", lambda: seen.append("restore"))
    monkeypatch.setattr(
        rc, "run_cmd", lambda cmd, *a, **k: (seen.append(" ".join(cmd)), (0, "", ""))[1]
    )
    monkeypatch.setattr(rc, "_unstage_oversized_files", lambda: seen.append("SIZE_GATE"))
    monkeypatch.setattr(rc, "_unstage_mutation_proof_target", lambda: seen.append("MUTATION_GATE"))
    monkeypatch.setattr(rc, "PROJECT_ROOT", Path("/nonexistent-scopes-dir"))
    rc._stage_all_except_claimed()
    assert "SIZE_GATE" in seen
    assert seen.index("git add -A") < seen.index("SIZE_GATE") < seen.index("MUTATION_GATE")
