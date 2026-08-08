"""`_ensure_server` no longer leaks a still-loading llama-server child.

REQ-ARC-WMTE-6221 / SCENARIO-ARC-WMTE-6221-A-STALE-LIVE-CHILD-IS-TERMINATED-BEFORE-A-REPLACEMENT-IS-LAUNCHED
REQ-ARC-WMTE-6221 / SCENARIO-ARC-WMTE-6221-B-WAIT-EXHAUSTION-TERMINATES-THE-LOADING-CHILD-INSTEAD-OF-ORPHANING-IT

WHY THIS EXISTS (2026-08-08 adversarial review, "Correctness" section, finding 4).

`_ensure_server` Popens a llama-server and stores it on `self._proc`. Two of its own paths used
to replace that reference without stopping whatever was there first:

* On wait exhaustion (the server never answered `/health` within the retry budget), the method
  returned `False` and left the child running. The NEXT call, finding the port unhealthy, would
  Popen a second child on the same port and overwrite `self._proc` -- so the first child kept
  running with nothing left pointing at it.
* The "refused to reuse" path (wrong model, or a smaller context pool than required) relaunches
  on a fresh port without first checking whether `self._proc` was still a live process.

`stop()` can only ever terminate `self._proc`, so once it is overwritten the earlier process is
unreachable by ANY cleanup path in this class -- an orphan that can hold ~20 GB of VRAM with
nothing in the program able to stop it.

The fix adds `LocalGGUFProposer._terminate_stale_proc`, called right before every place that
would otherwise silently drop a live `self._proc`, plus once more on the wait-exhaustion return
itself. It also records the cleanup on `self.orphaned_child_cleanups` (surfaced in
`liveness_witness()`) so a leak that DOES happen is visible in the artifact, not just in
`nvidia-smi` on the host.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as m  # noqa: E402


class _FakeProc:
    """Doubles as the fake `subprocess.Popen` CALLABLE and the process handle it returns --
    same idiom as `_FakePopen` in test_arc_generator_stderr_capture.py, extended with the
    terminate/kill/wait surface `_terminate_stale_proc` actually exercises.

    `alive` and `dies_on_terminate` are both controllable so tests can build every combination
    the fix has to handle: a genuinely-live stale child, an already-exited one, and one that
    ignores SIGTERM and needs the SIGKILL escalation.
    """

    _next_pid = 500000
    instances: list = []

    def __init__(self, args=None, **kwargs):
        _FakeProc._next_pid += 1
        self.pid = _FakeProc._next_pid
        self.alive = True
        self.dies_on_terminate = True
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        _FakeProc.instances.append(self)

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminate_calls += 1
        if self.dies_on_terminate:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.alive:
            raise subprocess.TimeoutExpired(cmd="fake-llama-server", timeout=timeout)
        return 0


def _make_proposer(tmp_path, monkeypatch):
    """Drive the REAL `_ensure_server` far enough to reach the Popen call, with no real server --
    same stubbing shape as the `launched` fixture in test_arc_generator_stderr_capture.py."""
    monkeypatch.setenv("CARNOT_ARC_SERVER_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(m.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(m.time, "sleep", lambda _seconds: None)  # skip the real retry-loop delay
    fake_bin = tmp_path / "llama-server"
    fake_bin.write_text("#!/bin/true\n")
    fake_bin.chmod(0o755)
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(
        m, "_generator_server_and_env", lambda _ffn_cpu_layers=None, _mtp=None: (fake_bin, None)
    )
    p = m.LocalGGUFProposer()
    p.model_path = str(fake_gguf)
    p.port = 45101
    _FakeProc.instances = []
    return p


def _stub_healthy_after_launch(monkeypatch, proposer):
    """`_healthy()` is False on the pre-launch check, True from then on -- i.e. the launch
    succeeds. Used by tests that are not exercising the wait-exhaustion path itself."""
    calls = {"n": 0}

    def _healthy(self):
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(type(proposer), "_healthy", _healthy, raising=False)


def test_wait_exhaustion_terminates_the_child_and_clears_the_reference(tmp_path, monkeypatch):
    """THE NAMED DEFECT, half A: on wait exhaustion the never-healthy child must be terminated
    and `self._proc` cleared BEFORE `_ensure_server` returns -- not left running and referenced
    only by a `self._proc` about to be overwritten by the next call."""
    p = _make_proposer(tmp_path, monkeypatch)
    monkeypatch.setattr(type(p), "_healthy", lambda self: False, raising=False)  # never healthy

    ok = p._ensure_server()

    assert ok is False
    assert len(_FakeProc.instances) == 1, "exactly one server should have been launched"
    spawned = _FakeProc.instances[0]
    assert spawned.terminate_calls == 1, "the never-healthy child must be terminated"
    assert spawned.alive is False
    assert p._proc is None, (
        "the reference must be cleared, not left pointing at a dead/orphaned child"
    )
    assert p.orphaned_child_cleanups, "the leak must be recorded, not silent"
    assert "wait budget" in p.orphaned_child_cleanups[0]


def test_a_live_previous_child_is_terminated_before_a_replacement_is_launched(
    tmp_path, monkeypatch
):
    """THE NAMED DEFECT, half B: a live child left over from a prior attempt must be stopped
    BEFORE the new Popen call overwrites `self._proc` -- not dropped and left running."""
    p = _make_proposer(tmp_path, monkeypatch)
    stale = _FakeProc()
    p._proc = stale
    _FakeProc.instances = []  # `stale` itself is hand-built, not a real Popen call -- don't count it
    _stub_healthy_after_launch(monkeypatch, p)

    ok = p._ensure_server()

    assert ok is True
    assert stale.terminate_calls == 1, "the stale live child must be terminated before relaunch"
    assert stale.alive is False
    assert len(_FakeProc.instances) == 1, "only the replacement counts as a real Popen call"
    assert p._proc is _FakeProc.instances[0], "the reference must now point at the NEW child"
    assert p.orphaned_child_cleanups, "terminating a still-live prior child must be recorded"


def test_a_dead_previous_child_does_not_spuriously_call_terminate(tmp_path, monkeypatch):
    """A process that already exited on its own must not be touched again -- terminate()/kill()
    on an already-dead handle would be a no-op at best and a false leak record at worst."""
    p = _make_proposer(tmp_path, monkeypatch)
    dead = _FakeProc()
    dead.alive = False
    p._proc = dead
    _stub_healthy_after_launch(monkeypatch, p)

    ok = p._ensure_server()

    assert ok is True
    assert dead.terminate_calls == 0, "an already-exited process must not be terminated again"
    assert dead.kill_calls == 0
    assert p.orphaned_child_cleanups == [], "nothing was cleaned up, so nothing should be logged"


def test_no_previous_proc_is_a_clean_noop(tmp_path, monkeypatch):
    """The common case -- first-ever launch, `self._proc` is still `None` -- must not error or
    log a spurious cleanup entry."""
    p = _make_proposer(tmp_path, monkeypatch)
    assert p._proc is None
    _stub_healthy_after_launch(monkeypatch, p)

    ok = p._ensure_server()

    assert ok is True
    assert p.orphaned_child_cleanups == []


def test_kill_fallback_when_terminate_does_not_reap_in_time(tmp_path, monkeypatch):
    """If SIGTERM does not kill the child within the wait, SIGKILL must follow -- otherwise the
    process would still be orphaned, just with an extra no-op terminate() call first."""
    p = _make_proposer(tmp_path, monkeypatch)
    stubborn = _FakeProc()
    stubborn.dies_on_terminate = False
    p._proc = stubborn

    p._terminate_stale_proc("test cleanup")

    assert stubborn.terminate_calls == 1
    assert stubborn.kill_calls == 1, "must escalate to SIGKILL when terminate() does not work"
    assert stubborn.alive is False
    assert p._proc is None
