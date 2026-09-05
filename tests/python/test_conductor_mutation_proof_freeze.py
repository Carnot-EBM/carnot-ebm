"""REQ-INFRA-6977: a checkpoint must not publish a file under an open mutation proof.

INCIDENT 2026-09-05. A mutation proof breaks a tracked file on purpose, confirms the suite
goes RED, then restores it byte-identically. The conductor's checkpoint commits whatever is
dirty every few minutes and runs no hooks, so it published a deliberately-broken module to
main, where it stood for 3 minutes 47 seconds.

The decision rule is the pure function `mutation_frozen`, tested here without a git fixture
or a real lock, following the precedent of `claimed_by_other_sessions`. The tests below also
bite the CALL SITES: a rule that is correct and never invoked proves nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import research_conductor as rc  # noqa: E402
import test_suite_mutation_check as tsmc  # noqa: E402


# ---------------------------------------------------------------- the pure rule


def test_the_proof_target_is_withheld() -> None:
    assert rc.mutation_frozen(["a.py", "b.py"], "b.py") == ["b.py"]


def test_no_open_proof_withholds_nothing() -> None:
    """Inert without a lock. The conductor checkpoints constantly; a default-on
    narrowing would wedge the research loop."""
    assert rc.mutation_frozen(["a.py", "b.py"], None) == []


def test_a_target_that_is_not_staged_is_not_invented() -> None:
    assert rc.mutation_frozen(["a.py"], "untouched.py") == []


# ------------------------------------------------------- reading the real lock


@pytest.fixture()
def lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the OWNING module's lock constant at tmp_path.

    `open_mutation_proof_target` imports PROOF_LOCK at call time precisely so the path is
    never duplicated here -- the real lock lives under `--git-common-dir` and an
    `ops/...lock` copy in the conductor would miss it while still looking like a guard.
    """
    p = tmp_path / "carnot_mutation_proof.lock"
    monkeypatch.setattr(tsmc, "PROOF_LOCK", p)
    return p


def test_an_open_proof_names_its_target_repo_relative(lock: Path) -> None:
    lock.write_text(json.dumps({"target": str(REPO / "python" / "carnot" / "x.py")}))
    assert rc.open_mutation_proof_target() == "python/carnot/x.py"


def test_no_lock_file_means_no_target(lock: Path) -> None:
    assert not lock.exists()
    assert rc.open_mutation_proof_target() is None


def test_an_unreadable_lock_fails_open_and_says_so(
    lock: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Fail-open here, deliberately: the checkpoint exists to preserve in-flight work, and
    losing that is unrecoverable while a published broken file is one revert away. Loud,
    because a guard that is trusted and silent is the worst state in this system."""
    lock.write_text("{not json")
    with caplog.at_level("WARNING"):
        assert rc.open_mutation_proof_target() is None
    assert "could not be read" in caplog.text


def test_a_target_outside_this_checkout_needs_no_exclusion(lock: Path, tmp_path: Path) -> None:
    """A worktree's mutation cannot be staged from the main checkout."""
    lock.write_text(json.dumps({"target": str(tmp_path / "elsewhere" / "y.py")}))
    assert rc.open_mutation_proof_target() is None


# ------------------------------------------------------------- the call sites


def test_the_frozen_path_is_actually_unstaged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bites the call site: the rule must reach `git restore --staged`, not merely exist."""
    calls: list[list[str]] = []

    def fake(cmd, *a, **k):  # noqa: ANN001, ANN202
        calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--cached"]:
            return 0, "keep.py\nbroken.py\n", ""
        return 0, "", ""

    monkeypatch.setattr(rc, "run_cmd", fake)
    monkeypatch.setattr(rc, "open_mutation_proof_target", lambda: "broken.py")
    rc._unstage_mutation_proof_target()
    assert ["git", "restore", "--staged", "--", "broken.py"] in calls


def test_nothing_is_unstaged_when_no_proof_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(rc, "run_cmd", lambda cmd, *a, **k: (calls.append(cmd), (0, "", ""))[1])
    monkeypatch.setattr(rc, "open_mutation_proof_target", lambda: None)
    rc._unstage_mutation_proof_target()
    assert not any("restore" in c for c in calls)


def test_the_staging_path_invokes_the_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bites the second call site. `_stage_all_except_claimed` returns early when no session
    has declared a scope -- the usual case -- so the exclusion must run BEFORE that."""
    seen: list[str] = []
    monkeypatch.setattr(rc, "_restore_dropped_determinations", lambda: seen.append("restore"))
    monkeypatch.setattr(
        rc, "run_cmd", lambda cmd, *a, **k: (seen.append(" ".join(cmd)), (0, "", ""))[1]
    )
    monkeypatch.setattr(rc, "_unstage_mutation_proof_target", lambda: seen.append("EXCLUDE"))
    monkeypatch.setattr(rc, "PROJECT_ROOT", Path("/nonexistent-scopes-dir"))
    rc._stage_all_except_claimed()
    assert "EXCLUDE" in seen
    assert seen.index("git add -A") < seen.index("EXCLUDE")


def test_the_file_by_file_checkpoint_drops_the_frozen_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interrupted-run checkpoint stages file-by-file and never reaches the index-based
    exclusion, so it carries its own. Extracted from `research_step` so the decision is
    provable; the one line calling it there is not covered — see the helper's docstring."""
    monkeypatch.setattr(rc, "open_mutation_proof_target", lambda: "broken.py")
    assert rc.checkpoint_after_mutation_freeze(["keep.py", "broken.py"]) == ["keep.py"]


def test_the_checkpoint_is_unchanged_with_no_proof_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc, "open_mutation_proof_target", lambda: None)
    assert rc.checkpoint_after_mutation_freeze(["a.py", "b.py"]) == ["a.py", "b.py"]


def test_a_checkpoint_of_only_the_frozen_path_commits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails CLOSED here, unlike the lock read. If the broken file is the only dirty path
    there is no work to preserve, and the sole alternative is publishing it."""
    monkeypatch.setattr(rc, "open_mutation_proof_target", lambda: "broken.py")
    assert rc.checkpoint_after_mutation_freeze(["broken.py"]) == []
