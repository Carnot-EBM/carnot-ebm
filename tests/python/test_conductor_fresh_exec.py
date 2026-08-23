"""Tests for the conductor's fresh-source re-exec (REQ-CONDUCTOR-FRESHEXEC-1).

Origin: 2026-08-22 — the conductor ran code up to ~11.5 hours older than
HEAD; three commits changed scripts/research_conductor.py while an old
process kept running, and a human noticed by comparing timestamps.

These tests exercise _maybe_reexec_on_fresh_source with every collaborator
monkeypatched: no git call, no real execv, no tracked-state write. The
wiring test asserts the loop actually calls it (a check nothing calls is
the bug class).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_conductor as rc  # noqa: E402

# No skipif here, deliberately. A concurrent session added one while the
# conductor edit was mid-clobber (pre-commit stash-restore failure); with a
# skip, a future clobber of the fresh-exec block would turn this whole file
# silently green. Tests must run and assert — a missing mechanism is RED.


class _ExecRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, exe, argv):
        self.calls.append((exe, argv))
        raise RuntimeError("execv reached")  # execv never returns; simulate


@pytest.fixture
def reexec_env(monkeypatch, tmp_path):
    """Point every side effect at tmp_path and record exec/log calls."""
    recorder = _ExecRecorder()
    log_lines = []
    monkeypatch.setattr(rc.os, "execv", recorder)
    monkeypatch.setattr(rc, "REEXEC_STATE", tmp_path / "reexec_state.json")
    monkeypatch.setattr(
        rc, "log_step", lambda task, status, details="": log_lines.append((task, status, details))
    )
    source = tmp_path / "research_conductor.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(rc, "CONDUCTOR_SOURCE", source)
    startup_sha = "0" * 64  # startup hash differs from anything real
    monkeypatch.setattr(rc, "_STARTUP_SOURCE_SHA", startup_sha)
    return recorder, log_lines, source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_commit_triggers_execv(reexec_env, monkeypatch):
    """SCENARIO-CONDUCTOR-FRESHEXEC-1-FRESH-COMMIT."""
    recorder, log_lines, source = reexec_env
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: _sha(source))
    with pytest.raises(RuntimeError, match="execv reached"):
        rc._maybe_reexec_on_fresh_source()
    assert len(recorder.calls) == 1
    exe, argv = recorder.calls[0]
    assert exe == sys.executable
    assert argv[1] == str(source)
    assert any("re-exec" in t.lower() and s == "OK" for t, s, _ in log_lines)


def test_dirty_tree_does_not_execv(reexec_env, monkeypatch):
    """SCENARIO-CONDUCTOR-FRESHEXEC-1-DIRTY-TREE: HEAD moved but the disk
    file differs from HEAD (an edit in flight) -> no exec."""
    recorder, _, source = reexec_env
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: "f" * 64)
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []


def test_broken_commit_escalates_and_does_not_execv(reexec_env, monkeypatch):
    """SCENARIO-CONDUCTOR-FRESHEXEC-1-BROKEN-COMMIT."""
    recorder, log_lines, source = reexec_env
    source.write_text("def broken(:\n")
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: _sha(source))
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []
    assert any("does not compile" in t for t, s, _ in log_lines if s == "WARN")


def test_git_failure_keeps_running(reexec_env, monkeypatch):
    recorder, log_lines, _ = reexec_env
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: None)
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []
    assert log_lines == []


def test_unchanged_source_is_a_no_op(reexec_env, monkeypatch):
    recorder, log_lines, source = reexec_env
    monkeypatch.setattr(rc, "_STARTUP_SOURCE_SHA", _sha(source))
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: _sha(source))
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []
    assert log_lines == []


def test_exec_storm_guard_attempts_each_hash_once(reexec_env, monkeypatch, tmp_path):
    """Rule 2 exec-storm guard: a hash already attempted never re-execs."""
    recorder, _, source = reexec_env
    head = _sha(source)
    (tmp_path / "reexec_state.json").write_text(json.dumps({"last_attempt_sha": head}))
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: head)
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []


def test_attempt_recorded_before_exec(reexec_env, monkeypatch, tmp_path):
    """The storm guard's state lands BEFORE execv: if exec dies mid-flight
    the next iteration must not retry forever."""
    recorder, _, source = reexec_env
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: _sha(source))
    with pytest.raises(RuntimeError):
        rc._maybe_reexec_on_fresh_source()
    state = json.loads((tmp_path / "reexec_state.json").read_text())
    assert state["last_attempt_sha"] == _sha(source)


def _code_only(source: str) -> str:
    """Source minus full-line AND trailing comments. The first version of
    this helper dropped only full-line comments, and the wiring mutation
    `pass  # _maybe_reexec_on_fresh_source()` stayed GREEN — the same
    commented-out-call blind spot as adversarial-review finding 5
    (2026-08-22), one shape further along. Naive `#` split is fine here:
    a `#` inside a string would only over-strip, which errs toward RED."""
    lines = []
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append(code)
    return "\n".join(lines)


def test_loop_wires_the_reexec_check():
    """A check nothing calls is the bug class: the main loop invokes the
    re-exec check on every --loop iteration after the first."""
    source = _code_only(inspect.getsource(rc.main))
    assert "_maybe_reexec_on_fresh_source()" in source
    assert "iteration > 1" in source


def test_committed_sha_reads_head_not_worktree():
    """Rule 1: committed bytes only — the implementation must go through
    `git show HEAD:...`, never a working-tree read."""
    source = _code_only(inspect.getsource(rc._committed_conductor_sha))
    assert "HEAD:scripts/research_conductor.py" in source


def test_import_crashing_commit_does_not_execv(reexec_env, monkeypatch):
    """Adversarial-review K3: a commit that COMPILES but crashes at import
    must not be exec'd into — systemd would relaunch the same broken HEAD
    every 30s, converting a running-good process into an outage."""
    recorder, log_lines, source = reexec_env
    source.write_text("import module_that_does_not_exist_anywhere\n")  # compiles, import-crashes
    monkeypatch.setattr(rc, "_committed_conductor_sha", lambda: _sha(source))
    rc._maybe_reexec_on_fresh_source()
    assert recorder.calls == []
    assert any("does not import" in t for t, s, _ in log_lines if s == "WARN")


def test_stop_authority_receipt_reader_warns_when_stale(monkeypatch, tmp_path):
    """Adversarial-review S1: the authority's receipt has a READER. A
    stale receipt WARNs durably; a fresh one stays silent."""
    log_lines = []
    monkeypatch.setattr(
        rc, "log_step", lambda task, status, details="": log_lines.append((task, status, details))
    )
    stale = tmp_path / "state.json"
    stale.write_text("{}")
    import os as _os

    old = stale.stat().st_mtime - 3 * 3600
    _os.utime(stale, (old, old))
    monkeypatch.setattr(rc, "STOP_AUTHORITY_STATE", stale)
    monkeypatch.setattr(rc, "_stop_authority_warned_day", [])
    rc._check_stop_authority_receipt()
    assert any("Stop-authority receipt STALE" in t for t, s, _ in log_lines)
    # fresh receipt: silent
    log_lines.clear()
    _os.utime(stale, None)
    monkeypatch.setattr(rc, "_stop_authority_warned_day", [])
    rc._check_stop_authority_receipt()
    assert log_lines == []


def test_stop_authority_receipt_check_is_wired():
    source = _code_only(inspect.getsource(rc.research_step))
    assert "_check_stop_authority_receipt()" in source
